"""Post-training static quantization (PTQ) of the FP32 baseline to INT8 (deliverable 2).

Steps: load FP32 -> fuse Conv+BN(+ReLU) -> pick the INT8 backend and qconfig -> prepare
(inserts observers) -> calibrate on 1,000 training images -> convert to INT8 -> evaluate -> save.
Also runs a small study of accuracy vs calibration-set size.

Run:  python scripts/02_ptq.py
      python scripts/02_ptq.py --quick
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from src.data import get_calibration_dataset, get_datasets, make_loader
from src.metrics import file_size_mb
from src.models import load_fp32_checkpoint
from src.quant_utils import check_all_convs_quantized, pick_backend, save_int8
from src.train_utils import CHECKPOINTS, RESULTS, evaluate, load_json, save_json, set_seed



@torch.no_grad()
def calibrate(model, calib_loader):
    """Run the calibration images through the model so the observers record activation ranges.

    No labels, no gradients, no weight updates: this only collects min/max (histogram) statistics.
    """
    model.eval()
    for images, _ in calib_loader:
        model(images)


def quantize_ptq(fp32_model, calib_loader):
    """Return an INT8 copy of fp32_model using post-training static quantization."""
    model = copy.deepcopy(fp32_model).cpu().eval()
    backend = pick_backend()  # fbgemm if this PyTorch has it, otherwise onednn (see quant_utils)
    model.fuse_model(is_qat=False)
    model.qconfig = get_default_qconfig(backend)  # per-channel weights, histogram activations
    prepare(model, inplace=True)
    calibrate(model, calib_loader)
    convert(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-size", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    set_seed()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cpu = torch.device("cpu")

    fp32_model, ckpt = load_fp32_checkpoint(CHECKPOINTS / "resnet18_fp32.pth")
    fp32_model.eval()
    _, val_ds, test_ds = get_datasets(quick=args.quick)
    val_loader = make_loader(val_ds, 250, shuffle=False, num_workers=2)
    test_loader = make_loader(test_ds, 250, shuffle=False, num_workers=2)

    calib_size = 200 if args.quick else args.calib_size
    calib_loader = make_loader(get_calibration_dataset(calib_size), 100, shuffle=False, num_workers=0)
    print(f"Backend: {pick_backend()} | calibration images: {calib_size}")

    int8_model = quantize_ptq(fp32_model, calib_loader)
    print(int8_model)
    n_qconv = check_all_convs_quantized(int8_model)

    fp32_val, _ = evaluate(fp32_model, val_loader, cpu)
    fp32_test, _ = evaluate(fp32_model, test_loader, cpu)
    val_acc, _ = evaluate(int8_model, val_loader, cpu)
    test_acc, _ = evaluate(int8_model, test_loader, cpu)

    example = next(iter(test_loader))[0][:16]
    script_path = save_int8(int8_model, "resnet18_ptq_int8", example, ckpt["config"],
                            {"val_acc": val_acc, "test_acc": test_acc, "calib_size": calib_size})

    fp32_size = file_size_mb(CHECKPOINTS / "resnet18_fp32.pth")
    int8_size = file_size_mb(script_path)
    summary = {
        "backend": pick_backend(), "calib_size": calib_size, "quantized_conv_modules": n_qconv,
        "fp32_val_acc": fp32_val, "fp32_test_acc": fp32_test,
        "ptq_val_acc": val_acc, "ptq_test_acc": test_acc, "test_drop": fp32_test - test_acc,
        "fp32_size_mb": fp32_size, "ptq_size_mb": int8_size, "size_ratio": fp32_size / int8_size,
        "quick": args.quick,
    }
    save_json(summary, RESULTS / "ptq_results.json")

    # Small study: does a bigger calibration set help?
    sizes = [50, 100, 200] if args.quick else [100, 500, 1000, 2000]
    rows = []
    for n in sizes:
        loader = make_loader(get_calibration_dataset(n), 100, shuffle=False, num_workers=0)
        m = quantize_ptq(fp32_model, loader)
        v, _ = evaluate(m, val_loader, cpu)
        t, _ = evaluate(m, test_loader, cpu)
        rows.append({"calib_size": n, "val_acc": round(v, 2), "test_acc": round(t, 2)})
        print(f"calibration size {n:5d}: val {v:.2f}% | test {t:.2f}%")
    with open(RESULTS / "ptq_calibration_size.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== PTQ summary =====")
    print(f"FP32 test accuracy      : {fp32_test:.2f}%")
    print(f"PTQ INT8 test accuracy  : {test_acc:.2f}%  (drop {fp32_test - test_acc:.2f} points)")
    print(f"PTQ INT8 val accuracy   : {val_acc:.2f}%")
    print(f"File size FP32 / INT8   : {fp32_size:.2f} MB / {int8_size:.2f} MB  ({fp32_size / int8_size:.2f}x smaller)")
    print(f"Saved {script_path}")


if __name__ == "__main__":
    main()
