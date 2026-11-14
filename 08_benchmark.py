"""Automatic benchmark of all models on the 10,000 CIFAR-10 test images (deliverable 7).

No arguments needed. For every model it measures:
  * top-1 test accuracy and per-class accuracy (and saves every prediction);
  * CPU latency at batch size 1 with ONE thread: 50 warm-up runs, 300 timed runs on real test images.
    The whole latency pass is repeated 3 times, models in a fixed order, and the median of each
    statistic is kept so one noisy pass does not decide the result;
  * CPU throughput at batch size 128 with all threads;
  * file size, parameters, nonzero parameters, sparsity and MACs.
Rows: the five PyTorch models, the ONNX Runtime models that exist, and (bonus) GPU latency for
the three float models.

Outputs: results/benchmark_summary.csv, predictions.csv, per_class_accuracy.csv, system_info.json

Run:  python scripts/08_benchmark.py
      python scripts/08_benchmark.py --quick      (fewer runs, 1,000 test images)
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch

from src.data import CLASSES, dataset_source, get_test_dataset, make_loader
from src.infer_utils import MODEL_FILES, load_model
from src.metrics import (count_macs, count_nonzero_params, count_params, file_size_mb,
                         gzip_size_mb, latency_stats, weight_sparsity)
from src.models import load_fp32_checkpoint
from src.quant_utils import pick_backend
from src.train_utils import RESULTS, save_json, set_seed

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "onnx_models"

# (row name, kind, file). Order is fixed and is also the latency measuring order.
TORCH_ROWS = [("fp32", "torch"), ("ptq_int8", "torch"), ("qat_int8", "torch"),
              ("unstructured", "torch"), ("structured", "torch")]
TORCH_NAME = {"fp32": "fp32", "ptq_int8": "ptq", "qat_int8": "qat",
              "unstructured": "unstructured", "structured": "structured"}
ONNX_ROWS = {
    "onnx_fp32": ONNX_DIR / "resnet18_fp32.onnx",
    "onnx_unstructured": ONNX_DIR / "resnet18_unstructured.onnx",
    "onnx_structured": ONNX_DIR / "resnet18_structured.onnx",
    "onnx_qat_int8_qdq": ONNX_DIR / "resnet18_qat_int8_qdq.onnx",
    "onnx_ort_ptq_int8": ONNX_DIR / "resnet18_ort_ptq_int8.onnx",
}
GPU_ROWS = ["fp32", "unstructured", "structured"]
ALL_THREADS = os.cpu_count()

COLUMNS = ["model", "format", "device", "test_accuracy", "accuracy_drop_vs_fp32", "size_mb",
           "params_total", "params_nonzero", "sparsity_pct", "macs_m", "latency_mean_ms",
           "latency_std_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
           "throughput_img_s", "speedup_vs_fp32", "compressed_size_mb"]


# ------------------------------------------------------------------ helpers

def cpu_name():
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        name, mem, driver = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
        return {"gpu": name, "gpu_memory": mem, "driver": driver}
    except Exception as e:
        return {"gpu": f"unavailable ({e})"}


def write_system_info():
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        **gpu_info(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": __import__("torchvision").__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "dataset_source": dataset_source(),
        "quantized_engine": pick_backend(),
        "supported_quantized_engines": torch.backends.quantized.supported_engines,
        "seed": 42,
        "benchmark_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(info, RESULTS / "system_info.json")
    return info


def ort_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    # without this, idle ORT threads busy-wait and fight PyTorch's threads for the cores
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def predict_torch(model, batches):
    preds, confs = [], []
    with torch.no_grad():
        for x in batches:
            p = torch.softmax(model(x).float(), 1)
            c, i = p.max(1)
            preds.append(i.numpy())
            confs.append(c.numpy())
    return np.concatenate(preds), np.concatenate(confs)


def predict_onnx(session, batches):
    preds, confs = [], []
    for x in batches:
        logits = session.run(None, {"input": x.numpy()})[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        preds.append(p.argmax(1))
        confs.append(p.max(1))
    return np.concatenate(preds), np.concatenate(confs)


def time_runs(fn, inputs, warmup, runs, sync_cuda=False):
    """Return a list of per-call times in seconds."""
    with torch.no_grad():
        for i in range(warmup):
            fn(inputs[i % len(inputs)])
        if sync_cuda:
            torch.cuda.synchronize()
        times = []
        for i in range(runs):
            x = inputs[i % len(inputs)]
            start = time.perf_counter()
            fn(x)
            if sync_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return times


def throughput(fn, batch, n_batches, sync_cuda=False):
    """Images per second for repeated calls on one batch (after two warm-up calls)."""
    with torch.no_grad():
        fn(batch)
        fn(batch)
        if sync_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_batches):
            fn(batch)
        if sync_cuda:
            torch.cuda.synchronize()
        return n_batches * len(batch) / (time.perf_counter() - start)


def onnx_runner(session):
    """Wrap an ONNX Runtime session so it can be called like a model: runner(x)."""
    def run(x):
        return session.run(None, {"input": x})
    return run


def size_and_params(name):
    """Static numbers for a PyTorch row. INT8 rows use the fused FP32 model for params / MACs."""
    file = MODEL_FILES[TORCH_NAME[name]]
    info = {"size_mb": file_size_mb(file), "compressed_size_mb": gzip_size_mb(file)}
    if name in ("ptq_int8", "qat_int8"):
        float_model, _ = load_fp32_checkpoint(MODEL_FILES["fp32"])
        float_model.eval().fuse_model()  # BN folded into the convs, like the INT8 model
    else:
        float_model, _ = load_fp32_checkpoint(file)
        float_model.eval()
    info["params_total"] = count_params(float_model)
    info["params_nonzero"] = count_nonzero_params(float_model)
    info["sparsity_pct"] = 100 * weight_sparsity(float_model)
    info["macs_m"] = count_macs(float_model) / 1e6
    return info


def onnx_static(path, torch_twin):
    """Size from the file, params and MACs from the matching PyTorch row."""
    info = dict(torch_twin)
    info["size_mb"] = file_size_mb(path)
    info["compressed_size_mb"] = gzip_size_mb(path)
    return info


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.warmup, args.runs, args.repeats = 5, 30, 1

    set_seed()
    pick_backend()
    info = write_system_info()
    print(f"CPU: {info['cpu']} | GPU: {info.get('gpu')} | torch {torch.__version__} | ORT {ort.__version__}")
    busy = psutil.cpu_percent(interval=2)
    print(f"CPU load before benchmarking: {busy:.0f}%")
    if busy > 20:
        print("NOTE: the machine looks busy; close other heavy programs for cleaner latency numbers.")

    test_ds = get_test_dataset(1000 if args.quick else None)
    labels = np.asarray(test_ds.labels)
    # Preprocess all test images once. Doing it batch by batch between ONNX Runtime calls made
    # PyTorch's and ORT's thread pools compete and slowed ORT down by more than 20x.
    loader = make_loader(test_ds, 250, shuffle=False, num_workers=0)
    batches = [x for x, _ in loader]
    single_inputs = [test_ds[i][0].unsqueeze(0) for i in range(min(args.runs, len(test_ds)))]
    batch128 = torch.stack([test_ds[i][0] for i in range(128)])

    # ---- load everything once
    torch_models = {name: load_model(TORCH_NAME[name], "cpu") for name, _ in TORCH_ROWS}
    onnx_paths = {n: p for n, p in ONNX_ROWS.items() if p.exists()}
    rows = {}
    for name, _ in TORCH_ROWS:
        fmt = "TorchScript INT8" if name.endswith("int8") else "PyTorch FP32"
        rows[name] = {"model": name, "format": fmt, "device": "cpu", **size_and_params(name)}
    twin = {"onnx_fp32": "fp32", "onnx_unstructured": "unstructured", "onnx_structured": "structured",
            "onnx_qat_int8_qdq": "qat_int8", "onnx_ort_ptq_int8": "ptq_int8"}
    for name, path in onnx_paths.items():
        static = onnx_static(path, {k: rows[twin[name]][k] for k in
                                    ("params_total", "params_nonzero", "sparsity_pct", "macs_m")})
        rows[name] = {"model": name, "format": "ONNX Runtime", "device": "cpu", **static}

    # ---- accuracy and predictions (all threads, batch 250)
    torch.set_num_threads(ALL_THREADS)
    predictions = {"image_id": np.arange(len(labels)), "true_label": labels,
                   "true_name": [CLASSES[i] for i in labels]}
    per_class = {}
    for name in rows:
        t = time.time()
        if name in torch_models:
            pred, conf = predict_torch(torch_models[name], batches)
        else:
            pred, conf = predict_onnx(ort_session(onnx_paths[name], ALL_THREADS), batches)
        correct = pred == labels
        rows[name]["test_accuracy"] = 100 * correct.mean()
        predictions[f"{name}_pred"] = pred
        predictions[f"{name}_conf"] = np.round(conf, 5)
        predictions[f"{name}_correct"] = correct.astype(int)
        per_class[name] = [100 * correct[labels == c].mean() for c in range(10)]
        print(f"accuracy {name:20s} {rows[name]['test_accuracy']:.2f}%  ({time.time() - t:.0f} s)")

    # ---- latency: 1 thread, batch 1, repeated passes in a fixed order, median of each statistic
    torch.set_num_threads(1)
    sessions_1t = {n: ort_session(p, 1) for n, p in onnx_paths.items()}
    np_inputs = [x.numpy() for x in single_inputs]
    passes = {name: [] for name in rows}
    pass_loads = []
    for rep in range(1, args.repeats + 1):
        psutil.cpu_percent(interval=None)  # start a new load measurement window
        for name in rows:
            if name in torch_models:
                times = time_runs(torch_models[name], single_inputs, args.warmup, args.runs)
            else:
                times = time_runs(onnx_runner(sessions_1t[name]), np_inputs, args.warmup, args.runs)
            passes[name].append(latency_stats(times))
        # whole-machine CPU load during the pass; our own single thread is about 100/16 = 6 %
        pass_loads.append(psutil.cpu_percent(interval=None))
        print(f"latency pass {rep}/{args.repeats} done | average CPU load during the pass {pass_loads[-1]:.0f}%")
        if pass_loads[-1] > 20:
            print("NOTE: other programs used the CPU during this pass; latency numbers may be inflated.")
    for name in rows:
        for key in passes[name][0]:
            rows[name][key] = float(np.median([p[key] for p in passes[name]]))
        print(f"latency  {name:20s} p50 {rows[name]['latency_p50_ms']:.3f} ms | "
              f"p99 {rows[name]['latency_p99_ms']:.3f} ms")

    # ---- throughput: batch 128, all threads
    torch.set_num_threads(ALL_THREADS)
    n_batches = 3 if args.quick else 10
    for name in rows:
        if name in torch_models:
            rows[name]["throughput_img_s"] = throughput(torch_models[name], batch128, n_batches)
        else:
            runner = onnx_runner(ort_session(onnx_paths[name], ALL_THREADS))
            rows[name]["throughput_img_s"] = throughput(runner, batch128.numpy(), n_batches)

    # ---- bonus: GPU latency for the float models (batch 1, synchronised)
    if torch.cuda.is_available():
        gpu_inputs = [x.cuda() for x in single_inputs]
        gpu_batch = batch128.cuda()
        for name in GPU_ROWS:
            model = load_model(TORCH_NAME[name], "cuda")
            stats_list = [latency_stats(time_runs(model, gpu_inputs, args.warmup, args.runs, sync_cuda=True))
                          for _ in range(args.repeats)]
            row = dict(rows[name])
            row.update({"model": f"{name}_gpu", "device": "cuda"})
            for key in stats_list[0]:
                row[key] = float(np.median([s[key] for s in stats_list]))
            row["throughput_img_s"] = throughput(model, gpu_batch, n_batches, sync_cuda=True)
            rows[f"{name}_gpu"] = row
            print(f"GPU latency {name:13s} p50 {row['latency_p50_ms']:.3f} ms")

    # ---- derived columns
    fp32_acc = rows["fp32"]["test_accuracy"]
    for name, row in rows.items():
        row["accuracy_drop_vs_fp32"] = fp32_acc - row["test_accuracy"]
        ref = rows["fp32_gpu"] if row["device"] == "cuda" else rows["fp32"]
        row["speedup_vs_fp32"] = ref["latency_p50_ms"] / row["latency_p50_ms"]

    # ---- write files
    with open(RESULTS / "benchmark_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in COLUMNS})

    with open(RESULTS / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(predictions)
        writer.writerow(keys)
        for i in range(len(labels)):
            writer.writerow([predictions[k][i] for k in keys])

    with open(RESULTS / "per_class_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + list(per_class))
        for c in range(10):
            writer.writerow([CLASSES[c]] + [round(per_class[m][c], 2) for m in per_class])

    save_json({"warmup": args.warmup, "runs": args.runs, "repeats": args.repeats,
               "latency_threads": 1, "throughput_threads": ALL_THREADS, "throughput_batch": 128,
               "test_images": len(labels), "cpu_load_before_pct": busy,
               "cpu_load_during_latency_passes_pct": pass_loads, "quick": args.quick},
              RESULTS / "benchmark_settings.json")

    print("\n===== Benchmark summary (CPU, latency with 1 thread, batch 1) =====")
    print(f"{'model':20s} {'acc %':>6s} {'size MB':>8s} {'nonzero':>11s} {'MACs M':>7s} "
          f"{'p50 ms':>7s} {'p99 ms':>7s} {'img/s':>7s} {'speedup':>7s}")
    for row in rows.values():
        print(f"{row['model']:20s} {row['test_accuracy']:6.2f} {row['size_mb']:8.2f} "
              f"{row['params_nonzero']:11,} {row['macs_m']:7.1f} {row['latency_p50_ms']:7.3f} "
              f"{row['latency_p99_ms']:7.3f} {row['throughput_img_s']:7.0f} {row['speedup_vs_fp32']:7.2f}")
    print(f"\nWrote {RESULTS / 'benchmark_summary.csv'} and predictions.csv ({len(labels)} rows)")


if __name__ == "__main__":
    main()
