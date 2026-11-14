"""Make every figure for the report from the results files, plus console-capture images.

Figures go to results/figures/ (200 dpi). Console captures (the real log text of the key runs,
rendered as dark monospaced images) go to report/screenshots/console_*.png.

Run:  python scripts/09_make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
SHOTS = ROOT / "report" / "screenshots"
LOGS = RESULTS / "logs"

MAIN_MODELS = ["fp32", "ptq_int8", "qat_int8", "unstructured", "structured"]
LABELS = {"fp32": "FP32", "ptq_int8": "PTQ INT8", "qat_int8": "QAT INT8",
          "unstructured": "Unstructured\n(90% sparse)", "structured": "Structured"}
COLORS = {"fp32": "#4C72B0", "ptq_int8": "#DD8452", "qat_int8": "#C44E52",
          "unstructured": "#55A868", "structured": "#8172B3"}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print(f"saved {FIG / name}")


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")


# ------------------------------------------------------------------ training curves

def fig_baseline_curves():
    h = pd.read_csv(RESULTS / "baseline_history.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(h.epoch, h.train_loss, label="train", lw=2)
    a1.plot(h.epoch, h.val_loss, label="validation", lw=2)
    a1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Baseline loss")
    a1.legend()
    a2.plot(h.epoch, h.train_acc, label="train", lw=2)
    a2.plot(h.epoch, h.val_acc, label="validation", lw=2)
    a2.set(xlabel="epoch", ylabel="accuracy (%)", title="Baseline accuracy")
    a2.legend()
    save(fig, "01_baseline_curves.png")


def fig_qat_curve():
    h = pd.read_csv(RESULTS / "qat_history.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h.epoch, h.train_acc, "o-", label="train (fake-quant)", lw=2)
    ax.plot(h.epoch, h.val_acc, "s-", label="validation (fake-quant)", lw=2)
    bn = h[h.bn_frozen == 1].epoch.min()
    obs = h[h.observers_off == 1].epoch.min()
    if not np.isnan(bn):
        ax.axvline(bn - 0.5, color="gray", ls="--")
        ax.text(bn - 0.45, sum(ax.get_ylim()) / 2 + 0.6, "BN frozen", color="gray", fontsize=11, va="top")
    if not np.isnan(obs):
        ax.axvline(obs - 0.5, color="black", ls=":")
        ax.text(obs - 0.45, sum(ax.get_ylim()) / 2 - 0.2, "observers off", fontsize=11, va="top")
    ax.set(xlabel="QAT epoch", ylabel="accuracy (%)", title="QAT fine-tuning")
    ax.legend(loc="lower right")
    save(fig, "02_qat_curve.png")


# ------------------------------------------------------------------ main comparison

def main_rows():
    df = pd.read_csv(RESULTS / "benchmark_summary.csv")
    return df.set_index("model").loc[MAIN_MODELS].reset_index()


def fig_main_bars(df):
    x = np.arange(len(df))
    labels = [LABELS[m] for m in df.model]
    colors = [COLORS[m] for m in df.model]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(x, df.test_accuracy, color=colors)
    bar_labels(ax, bars, "{:.2f}")
    low = df.test_accuracy.min()
    ax.set_ylim(max(0, low - 3), 100)
    ax.set_xticks(x, labels)
    ax.set(ylabel="test accuracy (%)", title="Test accuracy (10,000 CIFAR-10 images)")
    save(fig, "03_accuracy.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(x, df.size_mb, color=colors)
    bar_labels(ax, bars, "{:.1f}")
    ax.set_xticks(x, labels)
    ax.set(ylabel="file size on disk (MB)", title="Model file size")
    save(fig, "04_size.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, df.latency_p50_ms, color=colors, label="p50 (median)")
    ax.errorbar(x, df.latency_p50_ms, yerr=[np.zeros(len(df)), df.latency_p99_ms - df.latency_p50_ms],
                fmt="none", ecolor="black", capsize=6, label="up to p99")
    # label above the p99 whisker so the number and the whisker do not overlap
    for xi, p50, p99 in zip(x, df.latency_p50_ms, df.latency_p99_ms):
        ax.annotate(f"{p50:.2f}", (xi, p99), ha="center", va="bottom", fontsize=11,
                    xytext=(0, 4), textcoords="offset points")
    ax.set_ylim(0, df.latency_p99_ms.max() * 1.15)
    ax.set_xticks(x, labels)
    ax.set(ylabel="latency per image (ms)", title="CPU latency, batch 1, 1 thread")
    ax.legend()
    save(fig, "05_latency.png")


# ------------------------------------------------------------------ pruning

def fig_unstructured():
    it = pd.read_csv(RESULTS / "unstructured_iterations.csv")
    steps = it[it.iteration != "final"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps.measured_sparsity_pct, steps.val_acc_before_ft, "o--", label="right after pruning", lw=2)
    ax.plot(steps.measured_sparsity_pct, steps.val_acc_after_ft, "s-", label="after fine-tuning", lw=2)
    final = it[it.iteration == "final"]
    if len(final):
        ax.plot(final.measured_sparsity_pct, final.val_acc_after_ft, "*", ms=16, color="black",
                label="after final fine-tune")
    ax.set(xlabel="weight sparsity (%)", ylabel="validation accuracy (%)",
           title="Iterative global magnitude pruning")
    ax.legend()
    save(fig, "06_unstructured_iterations.png")

    layers = pd.read_csv(RESULTS / "unstructured_layer_sparsity.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(layers)), layers.sparsity_pct, color="#55A868")
    ax.set_xticks(range(len(layers)), layers.layer, rotation=75, ha="right", fontsize=9)
    ax.axhline(90, color="black", ls="--", lw=1)
    ax.set(ylabel="sparsity (%)", title="Per-layer sparsity after global pruning (dashed line: 90% overall)")
    save(fig, "07_unstructured_layer_sparsity.png")


def fig_structured():
    sw = pd.read_csv(RESULTS / "structured_sweep.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    # left: full scale, so the drop right after pruning is visible
    left.plot(sw.macs_m, sw.val_acc_after_prune, "x--", ms=8, lw=2, label="right after pruning")
    left.plot(sw.macs_m, sw.val_acc_after_ft, "o-", ms=8, lw=2, label="after fine-tuning")
    left.set_ylim(0, 100)
    left.set(xlabel="MACs per image (millions)", ylabel="validation accuracy (%)",
             title="Before and after fine-tuning")
    left.legend(loc="lower right")
    # right: zoom on the fine-tuned models, labelled with ratio and CPU latency
    right.plot(sw.macs_m, sw.val_acc_after_ft, "o-", ms=8, lw=2, color="C1")
    for _, r in sw.iterrows():
        right.annotate(f"ratio {r.ratio}\n{r.cpu_p50_ms:.2f} ms", (r.macs_m, r.val_acc_after_ft),
                       xytext=(6, -30), textcoords="offset points", fontsize=10)
    low, high = sw.val_acc_after_ft.min(), sw.val_acc_after_ft.max()
    right.set_ylim(low - 1.0, high + 0.5)
    right.set_xlim(sw.macs_m.min() - 30, sw.macs_m.max() + 90)
    right.set(xlabel="MACs per image (millions)", ylabel="validation accuracy (%)",
              title="After fine-tuning (labels: CPU p50 latency)")
    save(fig, "08_structured_sweep.png")


def fig_ptq_calibration():
    c = pd.read_csv(RESULTS / "ptq_calibration_size.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(c.calib_size, c.val_acc, "o-", lw=2, label="validation")
    ax.plot(c.calib_size, c.test_acc, "s-", lw=2, label="test")
    ax.set_xscale("log")
    ax.set_xticks(c.calib_size, [str(v) for v in c.calib_size])
    ax.set(xlabel="calibration images", ylabel="INT8 accuracy (%)", title="PTQ accuracy vs calibration size")
    ax.legend()
    save(fig, "09_ptq_calibration_size.png")


# ------------------------------------------------------------------ per class

def confusion(true, pred):
    m = np.zeros((10, 10), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


def fig_confusions(df):
    pred = pd.read_csv(RESULTS / "predictions.csv")
    compressed = df[df.model != "fp32"].sort_values("test_accuracy", ascending=False)
    best = compressed.iloc[0].model
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, name in zip(axes, ["fp32", best]):
        m = confusion(pred.true_label, pred[f"{name}_pred"])
        ax.imshow(m, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(10), CLASSES)
        ax.set(xlabel="predicted class", ylabel="true class",
               title=f"{LABELS[name].replace(chr(10), ' ')} (acc {100 * (pred[name + '_correct']).mean():.2f}%)")
    save(fig, "10_confusion_matrices.png")
    return best


def fig_per_class():
    pc = pd.read_csv(RESULTS / "per_class_accuracy.csv")
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.16
    x = np.arange(10)
    for k, m in enumerate(MAIN_MODELS):
        ax.bar(x + (k - 2) * width, pc[m], width, label=LABELS[m].replace("\n", " "), color=COLORS[m])
    ax.set_xticks(x, pc["class"])
    ax.set_ylim(max(0, pc[MAIN_MODELS].values.min() - 5), 100)
    ax.set(ylabel="test accuracy (%)", title="Per-class test accuracy")
    ax.legend(ncol=5, fontsize=10, loc="lower center")
    save(fig, "11_per_class_accuracy.png")


# ------------------------------------------------------------------ console captures

def section(text, start_marker, max_lines=40):
    """Lines from start_marker to the end of the log (or the last max_lines if not found)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if start_marker in line:
            return lines[i:i + max_lines]
    return lines[-max_lines:]


def render_console(lines, name, title):
    lines = [title, ""] + [line[:140] for line in lines]
    longest = max(len(line) for line in lines)
    height = 0.4 + 0.175 * len(lines)  # about 12.6 pt per line at font size 10
    width = max(8, 0.09 * longest + 0.4)  # about 6 pt per monospace character
    fig = plt.figure(figsize=(width, height), facecolor="#1e1e1e")
    fig.text(0.01, 0.99, "\n".join(lines), family="monospace", fontsize=10, color="#d4d4d4",
             va="top", ha="left")
    SHOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(SHOTS / name, dpi=150, facecolor="#1e1e1e")
    plt.close(fig)
    print(f"saved {SHOTS / name}")


def deliverables_tree():
    lines = ["edge_assignment/"]
    for folder in ["checkpoints", "onnx_models", "scripts", "src", "results", "report", "docs"]:
        path = ROOT / folder
        if not path.exists():
            continue
        lines.append(f"  {folder}/")
        for f in sorted(path.iterdir()):
            if f.name in ("__pycache__", "intermediate", "logs", "figures", "screenshots"):
                lines.append(f"    {f.name}/ ...")
            elif f.is_file():
                lines.append(f"    {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
    for f in ["README.md", "requirements.txt", "run_all.py"]:
        if (ROOT / f).exists():
            lines.append(f"  {f}")
    return lines


def console_captures():
    def log(name):
        p = LOGS / f"{name}.log"
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    base = log("01_train_baseline").splitlines()
    render_console(base[:3] + ["..."] + base[-12:], "console_baseline.png",
                   "Console output captured from the run: 01_train_baseline.py (start and end)")
    render_console(section(log("02_ptq"), "Quantized conv modules") [:3] + ["..."] +
                   section(log("02_ptq"), "calibration size", 25),
                   "console_ptq.png", "Console output captured from the run: 02_ptq.py")
    render_console(log("03_qat").splitlines()[:16] + ["..."] + section(log("03_qat"), "===== QAT summary"),
                   "console_qat.png", "Console output captured from the run: 03_qat.py")
    render_console(section(log("04_unstructured_prune"), "Masks removed"),
                   "console_unstructured.png", "Console output captured from the run: 04_unstructured_prune.py")
    render_console(section(log("05_structured_prune"), "--- prune ratio 0.5", 6) + ["..."] +
                   section(log("05_structured_prune"), "===== Structured"),
                   "console_structured.png", "Console output captured from the run: 05_structured_prune.py")
    render_console(log("06_export_onnx").splitlines()[-10:], "console_onnx.png",
                   "Console output captured from the run: 06_export_onnx.py")
    render_console(section(log("08_benchmark"), "===== Benchmark summary"),
                   "console_benchmark.png", "Console output captured from the run: 08_benchmark.py")
    render_console(log("07_inference").splitlines()[:14], "console_inference.png",
                   "Console output captured from the run: 07_inference.py --model all --num-samples 8")
    render_console(deliverables_tree(), "console_tree.png", "Deliverables folder tree (generated listing)")


def main():
    fig_baseline_curves()
    fig_qat_curve()
    df = main_rows()
    fig_main_bars(df)
    fig_unstructured()
    fig_structured()
    fig_ptq_calibration()
    best = fig_confusions(df)
    fig_per_class()
    console_captures()
    print(f"Best compressed model by test accuracy (used for the second confusion matrix): {best}")


if __name__ == "__main__":
    main()
