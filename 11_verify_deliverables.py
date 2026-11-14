"""Automatic self-check of every deliverable. Prints PASS/FAIL per check, exits 1 on any failure.

Checks: files exist; every checkpoint loads and runs; benchmark accuracies match a fresh
recomputation on 1,000 test images; predictions.csv is complete and agrees with the summary;
the unstructured model has no masks left and its sparsity is real; the structured model is
smaller and has no all-zero channels; ONNX parity; INT8 files are smaller than FP32; every
number quoted in the report equals the value in the results files; the ZIP has nothing it
should not have.

Run:  python scripts/11_verify_deliverables.py
      python scripts/11_verify_deliverables.py --quick   (expects the smaller quick-mode results)
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.data import get_test_dataset, make_loader
from src.infer_utils import MODEL_FILES, load_model
from src.metrics import count_params, format_number, weight_sparsity
from src.models import load_fp32_checkpoint
from src.quant_utils import pick_backend

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS_FOR = {"fp32": "fp32", "ptq": "ptq_int8", "qat": "qat_int8",
               "unstructured": "unstructured", "structured": "structured"}

REQUIRED = [
    # 1 baseline
    "checkpoints/resnet18_fp32.pth", "scripts/01_train_baseline.py",
    # 2 PTQ
    "scripts/02_ptq.py", "checkpoints/resnet18_ptq_int8.pt",
    # 3 QAT
    "scripts/03_qat.py", "checkpoints/resnet18_qat_int8.pt",
    # 4 unstructured
    "scripts/04_unstructured_prune.py", "checkpoints/resnet18_unstructured.pth",
    "checkpoints/resnet18_unstructured.pth.gz",
    # 5 structured
    "scripts/05_structured_prune.py", "checkpoints/resnet18_structured.pth",
    # 6 inference
    "scripts/07_inference.py", "src/infer_utils.py",
    # 7 benchmarking
    "scripts/08_benchmark.py", "results/benchmark_summary.csv", "results/predictions.csv",
    "results/per_class_accuracy.csv",
    # 8 ONNX
    "scripts/06_export_onnx.py", "scripts/07b_onnx_inference.py", "onnx_models/resnet18_fp32.onnx",
    "onnx_models/resnet18_unstructured.onnx", "onnx_models/resnet18_structured.onnx",
    # 9 report
    "report/Technical_Report.docx", "report/SCREENSHOTS_TODO.md",
    # 10 README and the rest
    "README.md", "requirements.txt", "run_all.py", "docs/VIVA_NOTES.md",
    "results/unstructured_iterations.csv", "results/unstructured_layer_sparsity.csv",
    "results/structured_sweep.csv", "results/ptq_calibration_size.csv", "results/system_info.json",
    "results/baseline_history.csv", "results/qat_history.csv", "results/unstructured_history.csv",
    "results/structured_history.csv",
]

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def check_files():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    check("all deliverable files exist", not missing, f"missing: {missing}" if missing else f"{len(REQUIRED)} files")


def check_models_and_accuracy(summary, predictions):
    test = get_test_dataset(1000)
    loader = make_loader(test, 250, shuffle=False, num_workers=0)
    labels = np.asarray(test.labels)
    for name in MODEL_FILES:
        try:
            model = load_model(name, "cpu")
            preds = []
            with torch.no_grad():
                for x, _ in loader:
                    preds.append(model(x).argmax(1).numpy())
            preds = np.concatenate(preds)
        except Exception as e:
            check(f"{name}: loads and runs", False, str(e))
            continue
        check(f"{name}: loads and runs on a test batch", True)
        fresh = 100 * (preds == labels).mean()
        row = RESULTS_FOR[name]
        stored = 100 * predictions[f"{row}_correct"].values[:1000].mean()
        check(f"{name}: fresh accuracy on 1,000 images matches predictions.csv",
              abs(fresh - stored) <= 0.3, f"fresh {fresh:.2f}% vs stored {stored:.2f}%")


def check_predictions(summary, predictions, quick):
    expected = 1000 if quick else 10000
    check("predictions.csv has one row per test image", len(predictions) == expected,
          f"{len(predictions)} rows")
    cpu_rows = summary[summary.device == "cpu"]
    worst = 0.0
    for _, r in cpu_rows.iterrows():
        recomputed = 100 * predictions[f"{r.model}_correct"].mean()
        worst = max(worst, abs(recomputed - r.test_accuracy))
    check("accuracy recomputed from predictions.csv equals benchmark_summary.csv", worst < 0.01,
          f"largest difference {worst:.4f} points")


def check_unstructured(summary):
    ckpt = torch.load(MODEL_FILES["unstructured"], map_location="cpu", weights_only=False)
    leftover = [k for k in ckpt["state_dict"] if k.endswith(("_orig", "_mask"))]
    check("unstructured: no weight_orig / weight_mask left", not leftover, str(leftover[:3]))
    model, _ = load_fp32_checkpoint(MODEL_FILES["unstructured"])
    gz_model, _ = load_fp32_checkpoint(ROOT / "checkpoints" / "resnet18_unstructured.pth.gz")
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), gz_model.state_dict().values()))
    check("unstructured: gzip checkpoint holds the same weights", same)
    real = 100 * weight_sparsity(model)
    reported = float(summary.set_index("model").loc["unstructured", "sparsity_pct"])
    check("unstructured: reported sparsity is the real sparsity", abs(real - reported) < 0.01,
          f"real {real:.2f}% vs reported {reported:.2f}%")


def check_structured():
    base, _ = load_fp32_checkpoint(MODEL_FILES["fp32"])
    small, _ = load_fp32_checkpoint(MODEL_FILES["structured"])
    check("structured: fewer parameters than the baseline", count_params(small) < count_params(base),
          f"{count_params(small):,} vs {count_params(base):,}")
    zero_channels = 0
    for m in small.modules():
        if isinstance(m, torch.nn.Conv2d):
            zero_channels += int((m.weight.abs().sum(dim=(1, 2, 3)) == 0).sum())
    check("structured: no all-zero conv channels (real rebuild, not masking)", zero_channels == 0,
          f"{zero_channels} zero channels")


def check_onnx():
    images = torch.stack([get_test_dataset(32)[i][0] for i in range(32)])
    for name in ["fp32", "unstructured", "structured"]:
        model, _ = load_fp32_checkpoint(MODEL_FILES[name])
        model.eval()
        sess = ort.InferenceSession(str(ROOT / "onnx_models" / f"resnet18_{name}.onnx"),
                                    providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(images).numpy()
        out = sess.run(None, {"input": images.numpy()})[0]
        diff = float(np.abs(ref - out).max())
        same = bool((ref.argmax(1) == out.argmax(1)).all())
        check(f"ONNX parity {name}", diff < 1e-3 and same, f"max diff {diff:.2e}, same predictions {same}")


def check_sizes():
    fp32 = MODEL_FILES["fp32"].stat().st_size
    for name in ["ptq", "qat"]:
        size = MODEL_FILES[name].stat().st_size
        check(f"{name} INT8 file smaller than FP32", size < fp32, f"{size / 1e6:.2f} MB vs {fp32 / 1e6:.2f} MB")


def read_source(source):
    """'file.csv|row|column' or 'file.json|key' -> the raw value from the results file."""
    parts = source.split("|")
    path = RESULTS / parts[0]
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        key_col = df.columns[0]
        df[key_col] = df[key_col].astype(str)
        return df.set_index(key_col).loc[parts[1], parts[2]]
    with open(path) as f:
        value = json.load(f)
    for key in parts[1:]:
        value = value[key]
    return value


def check_report_numbers():
    from docx import Document

    numbers_path = RESULTS / "report_numbers.json"
    if not numbers_path.exists():
        check("report numbers file exists", False)
        return
    with open(numbers_path) as f:
        numbers = json.load(f)
    doc = Document(str(ROOT / "report" / "Technical_Report.docx"))
    text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            text += "\n" + " | ".join(c.text for c in row.cells)
    bad = []
    for key, item in numbers.items():
        value = float(read_source(item["source"]))
        if item.get("absolute"):
            value = abs(value)
        expected = format_number(value, item["decimals"])
        if item["text"] != expected or item["text"] not in text:
            bad.append(f"{key}: report '{item['text']}' vs file '{expected}'")
    check("every number quoted in the report equals the results files", not bad,
          f"{len(numbers)} numbers checked" if not bad else "; ".join(bad[:5]))


def check_zip():
    zips = list(ROOT.glob("*_*.zip"))
    if not zips:
        print("[SKIP] no ZIP built yet")
        return
    with zipfile.ZipFile(zips[0]) as z:
        names = z.namelist()
    banned = [n for n in names if any(p in n.split("/") for p in ("data", ".venv", "__pycache__", "intermediate"))
              or Path(n).name.upper().startswith("CLAUDE")]
    check(f"ZIP {zips[0].name} has no data/.venv/__pycache__/intermediate/CLAUDE.md", not banned, str(banned[:5]))
    ckpts = [n for n in names if "/checkpoints/" in n]
    check("ZIP has only the final checkpoints", len(ckpts) == 8, f"{len(ckpts)} checkpoint files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    pick_backend()
    torch.set_num_threads(4)

    check_files()
    summary = pd.read_csv(RESULTS / "benchmark_summary.csv")
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    check_predictions(summary, predictions, args.quick)
    check_models_and_accuracy(summary, predictions)
    check_unstructured(summary)
    check_structured()
    check_onnx()
    check_sizes()
    check_report_numbers()
    check_zip()

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
