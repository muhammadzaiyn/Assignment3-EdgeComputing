"""Export models to ONNX and check them against PyTorch (deliverable 8).

Must-have exports (plain float graphs, opset 17, dynamic batch axis):
    fp32, unstructured, structured  -> onnx_models/<name>.onnx
Each is checked with onnx.checker, and ONNX Runtime outputs are compared with PyTorch.

Best-effort INT8 exports (only after the float ones work):
    qat_int8      : the trained QAT fake-quant model exported as QuantizeLinear/DequantizeLinear
    ort_ptq_int8  : ONNX Runtime's own static quantizer (QDQ) applied to the FP32 ONNX with the
                    same 1,000 calibration images. This is a different quantizer from the PyTorch one.
Any failure in the best-effort part is written to results/onnx_export.json instead of crashing.

Run:  python scripts/06_export_onnx.py
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                   MovingAveragePerChannelMinMaxObserver, QConfig,
                                   disable_observer, get_default_qat_qconfig, prepare_qat)

from src.data import get_calibration_dataset, get_test_dataset, make_loader
from src.metrics import file_size_mb
from src.models import build_model, load_fp32_checkpoint
from src.quant_utils import pick_backend
from src.train_utils import CHECKPOINTS, RESULTS, save_json, set_seed

ONNX_DIR = Path(__file__).resolve().parents[1] / "onnx_models"
OPSET = 17
FLOAT_MODELS = {
    "fp32": "resnet18_fp32.pth",
    "unstructured": "resnet18_unstructured.pth",
    "structured": "resnet18_structured.pth",
}


def export(model, path, example):
    """Legacy TorchScript-based exporter, opset 17, dynamic batch size."""
    torch.onnx.export(
        model, example, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def ort_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def parity(model, path, images):
    """Max abs difference of logits and prediction agreement between PyTorch and ORT."""
    with torch.no_grad():
        ref = model(images).numpy()
    out = ort_session(path).run(None, {"input": images.numpy()})[0]
    return float(np.abs(ref - out).max()), float((ref.argmax(1) == out.argmax(1)).mean())


def export_float_models(images, report):
    for name, file in FLOAT_MODELS.items():
        model, _ = load_fp32_checkpoint(CHECKPOINTS / file)
        model.eval()  # QuantStub / DeQuantStub are identity in a float model, so the graph is plain float
        path = ONNX_DIR / f"resnet18_{name}.onnx"
        export(model, path, images[:1])
        max_diff, agree = parity(model, path, images)
        ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
        print(f"{name:13s} exported | checker OK | max |torch - ort| {max_diff:.2e} | "
              f"prediction agreement {100 * agree:.1f}% | {file_size_mb(path):.2f} MB")
        assert max_diff < 1e-3 and agree == 1.0, f"ONNX parity failed for {name}"
        report[name] = {"status": "ok", "path": str(path.relative_to(ONNX_DIR.parent)),
                        "max_abs_diff": max_diff, "pred_agreement": agree,
                        "size_mb": file_size_mb(path), "ops": ops}


def qat_export_qconfig():
    """Same ranges as the QAT qconfig used in training, but built from the plain FakeQuantize class.

    The default QAT qconfig uses FusedMovingAvgObsFakeQuantize, whose fused op has no ONNX
    export rule. The plain FakeQuantize class exports to QuantizeLinear/DequantizeLinear and
    stores the same scale/zero-point buffers, so the trained values load straight into it.
    """
    trained = get_default_qat_qconfig(pick_backend())
    act_fq, weight_fq = trained.activation(), trained.weight()
    act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=act_fq.quant_min,
                                 quant_max=act_fq.quant_max, dtype=act_fq.dtype, qscheme=act_fq.qscheme)
    weight = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                    quant_min=weight_fq.quant_min, quant_max=weight_fq.quant_max,
                                    dtype=weight_fq.dtype, qscheme=weight_fq.qscheme, ch_axis=0)
    return QConfig(activation=act, weight=weight)


def export_qat_qdq(images, report):
    ckpt = torch.load(CHECKPOINTS / "intermediate" / "qat_prepared.pth", map_location="cpu",
                      weights_only=False)
    model = build_model(ckpt["config"]).train()
    model.fuse_model(is_qat=True)
    model.qconfig = qat_export_qconfig()
    prepare_qat(model, inplace=True)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # observer min/max buffers can differ in name; scales, zero points and weights must load
    bad = [k for k in missing if k.endswith(("scale", "zero_point", "weight", "bias"))]
    assert not bad, f"QAT state did not load: {bad[:5]}"
    model.apply(disable_observer)
    model.eval()

    path = ONNX_DIR / "resnet18_qat_int8_qdq.onnx"
    export(model, path, images[:1])
    max_diff, agree = parity(model, path, images)
    ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
    print(f"qat_int8 QDQ   exported | ops include QuantizeLinear: {'QuantizeLinear' in ops} | "
          f"max diff vs fake-quant torch {max_diff:.2e} | agreement {100 * agree:.1f}%")
    report["qat_int8"] = {"status": "ok", "path": str(path.relative_to(ONNX_DIR.parent)),
                          "max_abs_diff_vs_fakequant": max_diff, "pred_agreement": agree,
                          "size_mb": file_size_mb(path), "ops": ops,
                          "note": "weights are stored as float + QDQ nodes; ORT may run them as float"}


class CalibrationReader:
    """Feeds the 1,000 calibration images to onnxruntime.quantization one batch at a time."""

    def __init__(self, num_images):
        loader = make_loader(get_calibration_dataset(num_images), 100, shuffle=False, num_workers=0)
        self.batches = iter([{"input": x.numpy()} for x, _ in loader])

    def get_next(self):
        return next(self.batches, None)


def export_ort_ptq(images, report, calib_size):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = ONNX_DIR / "resnet18_fp32.onnx"
    pre_path = ONNX_DIR / "_fp32_preprocessed.onnx"
    out_path = ONNX_DIR / "resnet18_ort_ptq_int8.onnx"
    quant_pre_process(str(fp32_path), str(pre_path))
    quantize_static(str(pre_path), str(out_path), CalibrationReader(calib_size),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre_path.unlink()
    onnx.checker.check_model(onnx.load(str(out_path)))

    fp32_model, _ = load_fp32_checkpoint(CHECKPOINTS / "resnet18_fp32.pth")
    max_diff, agree = parity(fp32_model.eval(), out_path, images)
    print(f"ort_ptq_int8   exported | {file_size_mb(out_path):.2f} MB | "
          f"agreement with FP32 torch {100 * agree:.1f}%")
    report["ort_ptq_int8"] = {"status": "ok", "path": str(out_path.relative_to(ONNX_DIR.parent)),
                              "max_abs_diff_vs_fp32": max_diff, "pred_agreement_vs_fp32": agree,
                              "size_mb": file_size_mb(out_path), "calib_size": calib_size,
                              "note": "ONNX Runtime PTQ (QDQ), a different quantizer from the PyTorch one"}


def record_failure(report, name, error):
    print(f"{name}: best-effort export FAILED: {type(error).__name__}: {error}")
    report[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-size", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    set_seed()
    ONNX_DIR.mkdir(exist_ok=True)
    images = torch.stack([get_test_dataset(64)[i][0] for i in range(64)])
    report = {"opset": OPSET, "torch": torch.__version__, "onnx": onnx.__version__,
              "onnxruntime": ort.__version__}

    export_float_models(images, report)

    # best-effort INT8 exports: record a failure instead of stopping the pipeline
    try:
        export_qat_qdq(images, report)
    except Exception as e:
        record_failure(report, "qat_int8", e)
    try:
        export_ort_ptq(images, report, 200 if args.quick else args.calib_size)
    except Exception as e:
        record_failure(report, "ort_ptq_int8", e)

    save_json(report, RESULTS / "onnx_export.json")
    print(f"Wrote {RESULTS / 'onnx_export.json'}")


if __name__ == "__main__":
    main()
