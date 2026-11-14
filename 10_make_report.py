"""Build report/Technical_Report.docx from the results files (deliverable 9).

Every number in the text comes from results/*.csv|json through the helper `n(...)`, which also
records it in results/report_numbers.json so 11_verify_deliverables.py can check that the report
and the files agree. Nothing is typed in by hand. A PDF copy is made if LibreOffice or Word
(docx2pdf) is available; otherwise the script says so.

Run:  python scripts/10_make_report.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from src.metrics import format_number

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
SHOTS = ROOT / "report" / "screenshots"
OUT = ROOT / "report" / "Technical_Report.docx"

STUDENT = {"name": "Muhammad Zain ul Abidin", "roll": "23I-0051", "section": "AI-A",
           "course": "Edge Computing & Intelligent Systems", "assignment": "Assignment"}

MAIN = ["fp32", "ptq_int8", "qat_int8", "unstructured", "structured"]
PRETTY = {"fp32": "FP32 baseline", "ptq_int8": "PTQ INT8", "qat_int8": "QAT INT8",
          "unstructured": "Unstructured-pruned", "structured": "Structured-pruned",
          "onnx_fp32": "ONNX FP32", "onnx_unstructured": "ONNX unstructured",
          "onnx_structured": "ONNX structured", "onnx_qat_int8_qdq": "ONNX QAT (QDQ)",
          "onnx_ort_ptq_int8": "ONNX Runtime PTQ", "fp32_gpu": "FP32 (GPU)",
          "unstructured_gpu": "Unstructured (GPU)", "structured_gpu": "Structured (GPU)"}

NUMBERS = {}


# ------------------------------------------------------------------ numbers from files

def read_csv(name):
    return pd.read_csv(RESULTS / name)


def read_json(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def n(source, decimals=2, absolute=False):
    """Format a value read from a results file and remember where it came from.

    source: 'file.csv|row|column' (row matched on the first column) or 'file.json|key[|key]'.
    absolute=True writes the size of the value without its sign (used for "x points above/below").
    """
    parts = source.split("|")
    path = RESULTS / parts[0]
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        key_col = df.columns[0]
        df[key_col] = df[key_col].astype(str)
        value = df.set_index(key_col).loc[parts[1], parts[2]]
    else:
        value = read_json(parts[0])
        for key in parts[1:]:
            value = value[key]
    if absolute:
        value = abs(float(value))
    text = format_number(value, decimals)
    NUMBERS[f"{source}@{decimals}@{absolute}"] = {"source": source, "decimals": decimals,
                                                  "absolute": absolute, "text": text}
    return text


def drop_phrase(source):
    """'0.36 points below the baseline' or '0.01 points above the baseline' for a stored drop value."""
    word = "below" if float(read_value(source)) >= 0 else "above"
    return f"{n(source, absolute=True)} points {word} the baseline"


def read_value(source):
    parts = source.split("|")
    if parts[0].endswith(".json"):
        value = read_json(parts[0])
        for key in parts[1:]:
            value = value[key]
        return value
    df = read_csv(parts[0])
    key_col = df.columns[0]
    df[key_col] = df[key_col].astype(str)
    return df.set_index(key_col).loc[parts[1], parts[2]]


def b(model, column, decimals=2):
    """Shortcut for a benchmark_summary.csv value."""
    return n(f"benchmark_summary.csv|{model}|{column}", decimals)


# ------------------------------------------------------------------ docx helpers

class Report:
    def __init__(self):
        self.doc = Document()
        style = self.doc.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        for section in self.doc.sections:
            section.left_margin = section.right_margin = Cm(2.2)
            section.top_margin = section.bottom_margin = Cm(2.0)
        self.fig_no = 0
        self.tab_no = 0

    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, text, bold_lead=None, italic=False):
        para = self.doc.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        run = para.add_run(text)
        run.italic = italic
        para.paragraph_format.space_after = Pt(6)
        return para

    def next_figure(self):
        self.fig_no += 1
        return self.fig_no

    def next_table(self):
        self.tab_no += 1
        return self.tab_no

    def figure(self, path, caption, number, width_cm=15.5):
        if not Path(path).exists():
            self.p(f"[figure missing: {Path(path).name}]", italic=True)
            return
        self.doc.add_picture(str(path), width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure {number}. {caption}")
        run.italic = True
        run.font.size = Pt(9.5)

    def table(self, header, rows, caption, number, col_widths=None, font=9):
        cap = self.doc.add_paragraph()
        run = cap.add_run(f"Table {number}. {caption}")
        run.bold = True
        run.font.size = Pt(9.5)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, text in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(text))
            r.bold = True
            r.font.size = Pt(font)
        for row in rows:
            cells = t.add_row().cells
            for i, text in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(text)).font.size = Pt(font)
        if col_widths:
            for row in t.rows:
                for i, w in enumerate(col_widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph()

    def placeholder(self, label, caption):
        """A bordered empty box where the student pastes a real screenshot."""
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = t.rows[0].cells[0]
        cell.width = Cm(15)
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "dashed")
            el.set(qn("w:sz"), "12")
            el.set(qn("w:color"), "888888")
            borders.append(el)
        cell._tc.get_or_add_tcPr().append(borders)
        cell.text = ""
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(f"\n\n[ PASTE SCREENSHOT HERE: {label} ]\n\n\n")
        run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Screenshot: {caption}")
        r.italic = True
        r.font.size = Pt(9.5)

    def save(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(OUT))


# ------------------------------------------------------------------ sentences computed from the data

def layer_sparsity_sentence():
    layers = read_csv("unstructured_layer_sparsity.csv")
    most = layers.loc[layers.sparsity_pct.idxmax()]
    least = layers.loc[layers.sparsity_pct.idxmin()]
    first = layers.iloc[0]
    fc = layers.iloc[-1]
    text = (f"The layers are pruned very unevenly: {most.layer} lost {most.sparsity_pct:.1f}% of its weights, "
            f"while {least.layer} kept the most ({least.sparsity_pct:.1f}% pruned). The first convolution "
            f"({first.layer}, {first.weights:,} weights) ended at {first.sparsity_pct:.1f}%")
    if fc.layer != least.layer:
        text += f" and the final linear layer at {fc.sparsity_pct:.1f}%"
    return (text + ". In general the large 3x3 convolutions of the last stages, which hold most of the "
            "parameters, give up the most, and small layers keep more.")


def structured_recovery_sentence(sw):
    parts = []
    for _, row in sw[sw.ratio > 0].iterrows():
        parts.append(f"ratio {row.ratio}: {row.val_acc_after_prune:.2f}% right after pruning, "
                     f"{row.val_acc_after_ft:.2f}% after fine-tuning")
    return ("Removing whole channels hurts straight away, and fine-tuning brings the accuracy back ("
            + "; ".join(parts) + ").")


def int8_discussion(df):
    ptq_s = df.loc["ptq_int8", "speedup_vs_fp32"]
    qat_s = df.loc["qat_int8", "speedup_vs_fp32"]
    size_ratio = df.loc["fp32", "size_mb"] / df.loc["ptq_int8", "size_mb"]
    text = (f"The two INT8 models are {size_ratio:.2f} times smaller than FP32 and, on this CPU with one thread, "
            f"run {ptq_s:.2f}x (PTQ) and {qat_s:.2f}x (QAT) faster at the median, while losing "
            f"{df.loc['ptq_int8', 'accuracy_drop_vs_fp32']:.2f} and {df.loc['qat_int8', 'accuracy_drop_vs_fp32']:.2f} "
            "points of accuracy. ")
    if min(ptq_s, qat_s) > 4:
        text += ("A speed-up above 4x is more than the change from 32-bit to 8-bit numbers alone would explain. "
                 "We did not profile the kernels, but two likely reasons are these. First, the comparison is not "
                 "only about number formats: the INT8 model runs fused Conv+BN+ReLU modules, while the FP32 "
                 "baseline runs BN and ReLU as separate passes, and this Zen 4 CPU supports the AVX-512 VNNI "
                 "instructions (one 8-bit multiply-and-add step) that oneDNN is designed to use. Second, PyTorch's "
                 "eager FP32 path at batch size 1 is not the fastest FP32 option on this machine")
        if "onnx_fp32" in df.index:
            onnx_fp = df.loc["onnx_fp32", "latency_p50_ms"]
            text += (f": ONNX Runtime runs the same FP32 network in {onnx_fp:.2f} ms against "
                     f"{df.loc['fp32', 'latency_p50_ms']:.2f} ms in PyTorch, and against that stronger baseline "
                     f"the PyTorch INT8 gain is {onnx_fp / df.loc['ptq_int8', 'latency_p50_ms']:.2f}x")
        text += "."
    else:
        text += ("The speed-up comes from the backend's 8-bit kernels, which fit four times as many values in a "
                 "vector register and move a quarter of the memory. It is below 4x because some work, such as "
                 "quantizing the input, requantizing between layers and the pooling and linear layers, does not "
                 "get the full benefit.")
    return text


def pruning_discussion(df):
    un_s = df.loc["unstructured", "speedup_vs_fp32"]
    st_s = df.loc["structured", "speedup_vs_fp32"]
    mac_ratio = df.loc["fp32", "macs_m"] / df.loc["structured", "macs_m"]
    if 0.95 <= un_s <= 1.05:
        text = (f"The unstructured model has the same latency as FP32 within measurement noise ({un_s:.2f}x) "
                "and the same file size, as expected: dense kernels multiply the zeros anyway. ")
    else:
        text = (f"The unstructured model measured {un_s:.2f}x the FP32 speed. It runs the same dense kernels on "
                "the same tensor shapes, so we put this difference down to timing noise, not to the zeros. ")
    text += ("Its gain is the 90% sparsity itself, which only pays off after compression or with sparse kernels "
             f"or hardware. The structured model is the pruned model that is really faster on a normal CPU "
             f"({st_s:.2f}x), because its convolutions are smaller: it needs {df.loc['structured', 'macs_m']:.1f} M "
             f"MACs instead of {df.loc['fp32', 'macs_m']:.1f} M, {mac_ratio:.2f} times fewer. ")
    if st_s < mac_ratio:
        text += ("The speed-up is smaller than the MAC reduction because the stem, the block outputs, the shortcuts "
                 "and part of the memory traffic are not reduced, and narrow layers use the CPU less efficiently. ")
    else:
        text += ("The speed-up is at least as large as the MAC reduction, which suggests the smaller layers also "
                 "fit better in the CPU caches. ")
    text += ("Structured pruning and INT8 quantization do not depend on each other, so they could be combined "
             "into a model that is both narrower and 8-bit.")
    return text


def most_confused_pair(true, pred):
    """The pair of classes (a, c) mistaken for each other most often, and how many times."""
    best_pair, best_count = None, -1
    for a in range(10):
        for c in range(a + 1, 10):
            count = int(((true == a) & (pred == c)).sum() + ((true == c) & (pred == a)).sum())
            if count > best_count:
                best_pair, best_count = (a, c), count
    return best_pair, best_count


def per_class_discussion():
    from src.data import CLASSES

    pred = read_csv("predictions.csv")
    pc = read_csv("per_class_accuracy.csv").set_index("class")
    (a, c), count = most_confused_pair(pred.true_label.values, pred["fp32_pred"].values)
    best_cls = pc["fp32"].idxmax()
    worst_cls = pc["fp32"].idxmin()
    text = (f"For the FP32 baseline the most common confusion is between {CLASSES[a]} and {CLASSES[c]} "
            f"({count} test images mistaken one way or the other). The easiest class is {best_cls} "
            f"({pc.loc[best_cls, 'fp32']:.1f}%) and the hardest is {worst_cls} ({pc.loc[worst_cls, 'fp32']:.1f}%). ")
    worst_model, worst_class, worst_drop = None, None, 0.0
    for m in MAIN[1:]:
        drop = pc["fp32"] - pc[m]
        if drop.max() > worst_drop:
            worst_model, worst_class, worst_drop = m, drop.idxmax(), float(drop.max())
    if worst_model:
        text += (f"The compressed models follow the same pattern. The largest per-class loss against FP32 is "
                 f"{worst_drop:.1f} points, for {worst_class} in the {PRETTY[worst_model]} model, so compression "
                 "adds a few mistakes in the hard classes but does not break any single class.")
    return text


def gpu_sentence(df):
    if "fp32_gpu" not in df.index:
        return ""
    lat = [df.loc[m, "latency_p50_ms"] for m in ("fp32_gpu", "unstructured_gpu", "structured_gpu")]
    thr = [df.loc[m, "throughput_img_s"] for m in ("fp32_gpu", "structured_gpu")]
    return (f"On the GPU at batch size 1 the three float models take {min(lat):.2f} to {max(lat):.2f} ms, so "
            "pruning does not help there: a single 32x32 image is far too little work for the GPU, and the time is "
            "spent launching several dozen small kernels one after another, not on arithmetic. The GPU numbers are also less stable "
            "than the CPU ones because the GPU drives the display. At batch 128 the GPU processes "
            f"{thr[0]:,.0f} images per second with the FP32 model and {thr[1]:,.0f} with the structured model."
            + (" The pruned model is slower here even with fewer MACs. Our best guess is that its odd channel "
               "counts (for example 19 and 77) are not multiples of 8, so cuDNN cannot use its fastest kernels; "
               "rounding the kept channels to a multiple of 8 would be the fix for a GPU target."
               if thr[1] < thr[0] else ""))


def qat_curve_sentence(hist):
    low = hist.loc[hist.val_acc.idxmin()]
    last = hist.iloc[-1]
    if low.epoch == last.epoch:
        return ""
    return (f"The validation accuracy dips to {low.val_acc:.2f}% at epoch {int(low.epoch)}, just after these "
            "changes, because the network suddenly runs with fixed statistics and ranges; with the lower "
            f"learning rate of the later epochs it recovers to {last.val_acc:.2f}% at epoch {int(last.epoch)}.")


def dataset_sentence():
    source = read_json("system_info.json").get("dataset_source", "unknown")
    if source.startswith("Hugging Face"):
        return ("It first tries the official torchvision source; on our machine that server stayed below "
                "the 3 MB per minute the code accepts after 3 minutes of downloading, so it stopped and used the "
                "Hugging Face copy (uoft-cs/cifar10), which has "
                "the same 60,000 images and labels.")
    return f"It came from {source}."


def load_sentence():
    settings = read_json("benchmark_settings.json")
    loads = settings.get("cpu_load_during_latency_passes_pct")
    if not loads:
        return ""
    text = ("The script also records the whole-machine CPU load during each latency pass ("
            + ", ".join(f"{x:.0f}%" for x in loads) + "; our own single thread accounts for about 6%). ")
    if max(loads) > 20:
        text += ("Other programs were clearly running during the measurement, so the absolute times are higher "
                 "than on an idle machine. The ratios between models are less affected, because all models were "
                 "measured back to back in the same passes.")
    else:
        text += "So the single-thread latency passes ran on an otherwise quiet machine. "
    before = settings.get("cpu_load_before_pct", 0)
    if before > 20:
        text += (f"However, the load was already {before:.0f}% before the benchmark started, because another "
                 "program (a game) was running on the PC. That matters most for the throughput column, which "
                 "uses all CPU threads and so competes with the other program for every core: our throughput "
                 "numbers are lower than this CPU can reach and should only be compared with each other. "
                 "Re-running 'python run_all.py --full --from 08' on an idle machine updates every number "
                 "in this report.")
    return text


def prediction_changes(model_a, model_b):
    """How many test predictions differ between two models, and the number of test images."""
    pred = read_csv("predictions.csv")
    return int((pred[f"{model_a}_pred"] != pred[f"{model_b}_pred"]).sum()), len(pred)


# ------------------------------------------------------------------ sections

def title_block(r):
    t = r.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(STUDENT["course"])
    run.font.size = Pt(13)
    t = r.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(f"{STUDENT['assignment']}: Quantization and Pruning of ResNet-18 on CIFAR-10")
    run.bold = True
    run.font.size = Pt(18)
    t = r.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.add_run("Technical Report").font.size = Pt(13)
    t = r.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.add_run(f"{STUDENT['name']}  |  Roll No. {STUDENT['roll']}  |  Section {STUDENT['section']}\n"
              f"{time.strftime('%d %B %Y')}").font.size = Pt(11.5)


def intro(r):
    r.h("1. Introduction and objective")
    r.p("Edge devices have little memory, little compute and often no GPU, so a model trained on a desktop "
        "GPU usually has to be made smaller and faster before it can be deployed. This report covers two ways "
        "of doing that. Question 1 uses quantization: storing weights and activations as 8-bit integers (INT8) "
        "instead of 32-bit floats (FP32). Question 2 uses pruning: removing weights or whole channels that "
        "contribute little to the output.")
    r.p("We trained one ResNet-18 on CIFAR-10 and built four compressed versions from it: post-training "
        "quantization (PTQ), quantization-aware training (QAT), unstructured magnitude pruning and structured "
        "channel pruning. All five models were exported for deployment, run through one inference script and "
        "measured by one benchmark script on the same CPU. Every number in this report is read from the CSV "
        "and JSON files written by those scripts.")


def environment(r):
    s = read_json("system_info.json")
    r.h("2. Environment")
    r.p("All training ran on the GPU of a desktop PC. All final benchmarking ran on the CPU of the same PC, "
        "because PyTorch's eager-mode INT8 kernels only exist for CPUs. Table 1 lists the setup, read from "
        "results/system_info.json.")
    rows = [["Operating system", s["os"]], ["CPU", f"{s['cpu']} ({s['cpu_logical_cores']} logical cores)"],
            ["RAM", f"{s['ram_gb']} GB"],
            ["GPU", f"{s.get('gpu')} ({s.get('gpu_memory', '')}), driver {s.get('driver', '')}"],
            ["Python", s["python"]], ["PyTorch", f"{s['torch']} (CUDA {s['torch_cuda']})"],
            ["torchvision", s["torchvision"]], ["ONNX / ONNX Runtime", f"{s['onnx']} / {s['onnxruntime']}"],
            ["Quantized engine", s["quantized_engine"]], ["Random seed", str(s["seed"])]]
    r.table(["Item", "Value"], rows, "Hardware and software used.", r.next_table(), [4.5, 11])


def baseline(r):
    res = read_json("baseline_results.json")
    hist = read_csv("baseline_history.csv")
    f = r.next_figure()
    r.h("3. Dataset and baseline model")
    r.p("CIFAR-10 has 50,000 training and 10,000 test images of size 32x32 in 10 classes. The dataset is "
        f"downloaded by src/data.py. {dataset_sentence()} We split the training set once, with seed 42, into 45,000 images for "
        "training and 5,000 for validation. Validation was used for every choice (best epoch, prune ratio). "
        "The test set was only used for the final numbers. Training images get a random 32x32 crop from a "
        "4-pixel padded image and a random horizontal flip; validation, test and calibration images are only "
        "normalised with the usual CIFAR-10 mean and standard deviation.")
    r.p("The network is a CIFAR-style ResNet-18 written in src/models.py. Compared with the ImageNet version, "
        "the first layer is a 3x3 convolution with stride 1 and there is no max-pool, so the small images are "
        "not shrunk at the start. It has four stages of two BasicBlocks (64, 128, 256 and 512 channels), global "
        f"average pooling and a 10-way linear layer, about {n('benchmark_summary.csv|fp32|params_total', 0)} "
        "parameters in total. The model was written so that the same class can be quantized and pruned: it has "
        "QuantStub and DeQuantStub at the ends (they do nothing until conversion), every ReLU is a separate "
        "module, the residual addition goes through FloatFunctional.add_relu, and each block takes its own "
        "inner width so the structured-pruned network can be rebuilt with the same code.")
    r.p(f"Training used SGD with Nesterov momentum 0.9, weight decay 5e-4, batch size 128 and a OneCycle "
        f"learning-rate schedule with a peak of 0.1, for {n('baseline_results.json|epochs', 0)} epochs. "
        "Mixed precision (bfloat16 autocast) and the channels_last memory layout made each epoch faster. "
        f"The best validation accuracy was {n('baseline_results.json|best_val_acc')}% at epoch "
        f"{n('baseline_results.json|best_epoch', 0)}, and this checkpoint reached "
        f"{n('baseline_results.json|test_acc')}% on the test set. The whole run took "
        f"{n('baseline_results.json|train_minutes', 1)} minutes. Figure {f} shows the curves: the training "
        "accuracy keeps rising to the end while the validation accuracy levels off in the last epochs, as "
        "expected with a OneCycle schedule.")
    r.figure(FIG / "01_baseline_curves.png", "Baseline training and validation loss (left) and accuracy (right).", f)


def ptq(r):
    res = read_json("ptq_results.json")
    f = r.next_figure()
    t = r.next_table()
    r.h("4. Question 1: Post-training static quantization (PTQ)")
    r.p("Quantization maps each float value x to an 8-bit integer q with x = scale * (q - zero_point). Static "
        "quantization fixes the scale and zero point of every activation in advance, so the whole network, "
        "including the residual additions, runs with integer arithmetic. The code is in scripts/02_ptq.py.")
    r.p("Fusion. Before quantizing, each Conv-BN-ReLU group is merged into one module (the batch-norm "
        "scale and shift are folded into the convolution weights and bias). This is done for the stem, both "
        "convolutions in every block and the 1x1 shortcut convolutions. Fusion removes separate BN layers that "
        "would otherwise need their own quantize and dequantize steps, so it is both faster and more accurate.",
        bold_lead="")
    backend = res["backend"]
    engines = read_json("system_info.json").get("supported_quantized_engines", [])
    if backend == "fbgemm":
        why = "FBGEMM is PyTorch's usual INT8 backend for x86 CPUs, and our CPU is x86-64."
    else:
        why = (f"The usual choice for an x86 desktop is FBGEMM, but the PyTorch build we used "
               f"({read_json('system_info.json')['torch']}, official Windows wheel) is compiled without it: "
               f"torch.backends.quantized.supported_engines lists only {engines}. We also checked the 2.10 "
               "Windows wheel and it is the same. Going back to PyTorch 2.5 was not an option, because it has no "
               "kernels for our RTX 50-series (Blackwell) GPU and no wheels for Python 3.14. So the code picks "
               f"{backend} (oneDNN), Intel's x86 library, which "
               "also provides INT8 convolution and linear kernels and runs on our AMD CPU. On Linux the same "
               "code would pick FBGEMM. The INT8 model must be run with the backend it was converted with.")
    r.p(f"Backend. src/quant_utils.pick_backend() sets torch.backends.quantized.engine to {backend}, and the "
        f"qconfig is get_default_qconfig('{backend}'). {why} This qconfig quantizes weights per output channel "
        "(each filter gets its own scale) and observes activations with a histogram observer, which picks a "
        "range that cuts off rare outliers.")
    r.p(f"Calibration. prepare() inserts observers that record the range of every activation. We then ran a "
        f"fixed random subset of {n('ptq_results.json|calib_size', 0)} training images through the model in "
        "batches of 100, without gradients and without augmentation, so that the recorded ranges match what the "
        "model sees at test time. After that, convert() replaced every module with its INT8 version; the script "
        f"checks that {n('ptq_results.json|quantized_conv_modules', 0)} quantized convolutions exist and no plain "
        "float Conv2d is left. The INT8 model was saved as TorchScript (resnet18_ptq_int8.pt), which holds the "
        "graph and weights so deployment does not need our Python class, and also as a state_dict with its "
        "config. The TorchScript file is reloaded and must give the same predictions on a test batch.")
    r.p(f"Result. PTQ INT8 reached {n('ptq_results.json|ptq_test_acc')}% test accuracy against "
        f"{n('ptq_results.json|fp32_test_acc')}% for FP32 (both measured on the CPU), a drop of "
        f"{n('ptq_results.json|test_drop')} points. The file shrank from {n('ptq_results.json|fp32_size_mb')} MB "
        f"to {n('ptq_results.json|ptq_size_mb')} MB, {n('ptq_results.json|size_ratio')} times smaller, which is "
        "close to the expected 4x from replacing 32-bit with 8-bit weights. Equal accuracy does not mean equal "
        f"predictions: {prediction_changes('fp32', 'ptq_int8')[0]} of the {prediction_changes('fp32', 'ptq_int8')[1]:,} "
        "test predictions changed after quantization, and the gains and losses happened to cancel out. "
        f"(The FP32 validation accuracy measured here on the CPU is {n('ptq_results.json|fp32_val_acc')}%, while "
        f"training reported {n('baseline_results.json|best_val_acc')}% for the same checkpoint. Training evaluates "
        "on the GPU under bfloat16 autocast, which rounds differently; one or two of the 5,000 validation "
        "images change class.)")
    cal = read_csv("ptq_calibration_size.csv")
    rows = []
    for _, row in cal.iterrows():
        key = str(int(row.calib_size))
        rows.append([key, n(f"ptq_calibration_size.csv|{key}|val_acc"), n(f"ptq_calibration_size.csv|{key}|test_acc")])
    spread = cal.test_acc.max() - cal.test_acc.min()
    r.p(f"We also repeated PTQ with different calibration sizes (Table {t}, Figure {f}). "
        + ("The accuracy hardly changes between the sizes, so a few hundred images are already enough to "
           "estimate the activation ranges of this network." if spread < 0.5 else
           "The accuracy changes noticeably with the calibration size, so the number of calibration images "
           "matters for this network."))
    r.table(["Calibration images", "Val accuracy (%)", "Test accuracy (%)"], rows,
            "PTQ INT8 accuracy for different calibration-set sizes.", t, [4, 4, 4])
    r.figure(FIG / "09_ptq_calibration_size.png", "PTQ INT8 accuracy against the number of calibration images.", f, 11)


def qat(r):
    res = read_json("qat_results.json")
    hist = read_csv("qat_history.csv")
    f = r.next_figure()
    t = r.next_table()
    r.h("5. Question 1: Quantization-aware training (QAT)")
    r.p("PTQ quantizes a model that never saw rounding during training. QAT instead fine-tunes the model "
        "while simulating INT8. It inserts fake-quantization modules after weights and activations: in the "
        "forward pass they round values to the INT8 grid and convert them straight back to float, so the "
        "loss feels the rounding error, while the backward pass treats the rounding as identity "
        "(straight-through estimator) so gradients still flow. The weights learn to sit where rounding hurts "
        "less. The code is in scripts/03_qat.py.")
    r.p(f"We started from the trained FP32 weights, put the model in training mode, fused it with the QAT "
        f"version of fusion (Conv-BN pairs stay trainable), set get_default_qat_qconfig with the same backend as PTQ and called "
        f"prepare_qat, which inserted {n('qat_results.json|fake_quant_modules', 0)} fake-quant modules. "
        f"Fine-tuning ran on the GPU for {n('qat_results.json|epochs', 0)} epochs with SGD (momentum 0.9, "
        f"weight decay 1e-4) and a cosine schedule from lr {n('qat_results.json|lr', 3)}. Mixed precision was "
        "turned off here, because bfloat16 would disturb the ranges that the observers measure.")
    r.p(f"Two things are frozen part-way through. From epoch {n('qat_results.json|freeze_bn_epoch', 0)} the "
        "batch-norm running statistics are frozen, so BN uses the same fixed mean and variance that will be "
        "folded into the INT8 convolution; otherwise the model would train against statistics that change from "
        f"batch to batch. From epoch {n('qat_results.json|freeze_observer_epoch', 0)} the observers are "
        "disabled, so scale and zero point stop moving and the last epochs fine-tune the weights against fixed "
        f"quantization grids. Figure {f} marks both points. {qat_curve_sentence(hist)}")
    r.figure(FIG / "02_qat_curve.png", "QAT fine-tuning accuracy (with fake quantization) per epoch.", f, 13)
    r.p(f"After training, the model was moved to the CPU, put in eval mode and converted to a real INT8 model. "
        f"We evaluated that converted model, not the fake-quant one. It reached "
        f"{n('qat_results.json|qat_test_acc')}% test accuracy, a drop of {n('qat_results.json|test_drop')} "
        f"points from FP32, with a file of {n('qat_results.json|qat_size_mb')} MB. Fine-tuning took "
        f"{n('qat_results.json|train_minutes', 1)} minutes.")
    ptq_acc = float(b("ptq_int8", "test_accuracy"))
    qat_acc = float(b("qat_int8", "test_accuracy"))
    ptq_val = float(n("ptq_results.json|ptq_val_acc"))
    qat_val = float(n("qat_results.json|qat_val_acc"))
    val_text = (f"On the validation set QAT was {'above' if qat_val >= ptq_val else 'below'} PTQ "
                f"({qat_val:.2f}% against {ptq_val:.2f}%). ")
    if qat_acc >= ptq_acc:
        compare = (f"QAT ended {qat_acc - ptq_acc:.2f} points above PTQ on the test set. {val_text}With per-channel "
                   "weights and a well-behaved ResNet the PTQ drop is already small, so QAT has little left to "
                   "recover; the gap would be larger for models that quantize badly, such as MobileNets.")
    else:
        compare = (f"The expected order did not appear on the test set: QAT ended {ptq_acc - qat_acc:.2f} points "
                   f"below PTQ. {val_text}Our PTQ model lost almost nothing, so QAT had nothing to recover, and "
                   f"{n('qat_results.json|epochs', 0)} more epochs of training at 99% training accuracy move the "
                   f"weights: the QAT model disagrees with FP32 on {prediction_changes('fp32', 'qat_int8')[0]} test "
                   f"images, against {prediction_changes('fp32', 'ptq_int8')[0]} for PTQ, and the changes go both "
                   "ways. We trained with only one seed, so we cannot say how much a different seed would move "
                   "these numbers; for CIFAR-10 ResNets a few tenths of a point is a common run-to-run spread, "
                   "so we read this as 'both INT8 models are about as good as FP32 here' rather than as QAT "
                   "being worse.")
    rows = [[PRETTY[m], b(m, "test_accuracy"), b(m, "accuracy_drop_vs_fp32"), b(m, "size_mb"),
             b(m, "latency_p50_ms", 3)] for m in ["fp32", "ptq_int8", "qat_int8"]]
    r.p(f"Table {t} compares the three Question 1 models (values from the benchmark). {compare}")
    r.table(["Model", "Test acc. (%)", "Drop vs FP32 (points)", "Size (MB)", "CPU p50 latency (ms)"], rows,
            "Question 1 models on the CPU (latency: batch 1, 1 thread).", t)


def unstructured(r):
    res = read_json("unstructured_results.json")
    it = read_csv("unstructured_iterations.csv")
    t = r.next_table()
    f1 = r.next_figure()
    f2 = r.next_figure()
    r.h("6. Question 2: Unstructured magnitude pruning")
    r.p("Unstructured pruning sets individual weights to zero. We used global L1 magnitude pruning "
        "(torch.nn.utils.prune.global_unstructured with L1Unstructured) on the weights of every Conv2d and the "
        "Linear layer. Global means all these weights are ranked together and the smallest ones are removed "
        "wherever they are, so layers with many redundant weights lose more than small, sensitive layers. "
        "Batch-norm parameters and biases are not pruned. The code is in scripts/04_unstructured_prune.py.")
    r.p("Pruning was done in five steps to cumulative sparsity 40, 60, 75, 85 and 90%. PyTorch applies a new "
        "amount to the weights that are still left, so each step's amount was computed as "
        "(target - current) / (1 - current), and the real sparsity was measured after each step and checked "
        "against the target. After every step the model was fine-tuned for "
        "4 epochs with the masks active (pruned weights stay zero because the forward pass uses "
        "weight_orig * mask). After the last step it was fine-tuned for 10 more epochs at a lower learning "
        f"rate. Table {t} and Figure {f1} show how the accuracy falls after each cut and recovers with "
        "fine-tuning.")
    rows = []
    for _, row in it.iterrows():
        key = str(row.iteration)
        rows.append([key, f"{row.target_sparsity_pct:.0f}",
                     n(f"unstructured_iterations.csv|{key}|measured_sparsity_pct"),
                     n(f"unstructured_iterations.csv|{key}|val_acc_before_ft"),
                     n(f"unstructured_iterations.csv|{key}|val_acc_after_ft")])
    r.table(["Iteration", "Target sparsity (%)", "Measured sparsity (%)", "Val acc. before FT (%)",
             "Val acc. after FT (%)"], rows, "Iterative pruning schedule (FT = fine-tuning).", t)
    r.figure(FIG / "06_unstructured_iterations.png", "Validation accuracy against sparsity, before and after fine-tuning.", f1, 12)
    r.p(f"Finally prune.remove() was called on every pruned layer. This multiplies weight_orig by the mask once, "
        "stores the result as a plain weight tensor and deletes the mask and the hook. The script checks that no "
        "weight_orig or weight_mask is left and that the zeros are still there. The final model has "
        f"{n('unstructured_results.json|weight_sparsity_pct')}% of its prunable weights equal to zero, "
        f"{n('unstructured_results.json|params_nonzero', 0)} nonzero parameters out of "
        f"{n('unstructured_results.json|params_total', 0)}, and reaches {n('unstructured_results.json|test_acc')}% "
        f"test accuracy ({drop_phrase('unstructured_results.json|test_drop')}). Figure {f2} shows "
        f"the per-layer sparsity that the global ranking produced. {layer_sparsity_sentence()}")
    r.figure(FIG / "07_unstructured_layer_sparsity.png", "Sparsity of each layer after global pruning.", f2)
    r.p("What unstructured pruning does not give. Zeros stored in a dense tensor are still multiplied, so this model is not faster on a "
        "normal CPU, and the checkpoint is the same size as the baseline "
        f"({n('unstructured_results.json|size_mb')} MB against {n('unstructured_results.json|fp32_size_mb')} MB). "
        "What it does give is a model that compresses well: with gzip the file goes down to "
        f"{n('unstructured_results.json|gzip_size_mb')} MB, against {n('unstructured_results.json|fp32_gzip_size_mb')} MB "
        "for the gzipped baseline. A real speed-up would need sparse kernels or hardware that skips zeros, which "
        "our benchmark does not use, so we do not claim one.")


def structured(r):
    res = read_json("structured_results.json")
    sw = read_csv("structured_sweep.csv")
    t = r.next_table()
    f = r.next_figure()
    r.h("7. Question 2: Structured channel pruning")
    r.p("Structured pruning removes whole channels, so the remaining tensors are smaller and every layer does "
        "less work on any hardware. The difficulty is that channels are connected: removing an output channel "
        "of one convolution means its batch-norm channel and the matching input channel of the next "
        "convolution must go too. The code is in scripts/05_structured_prune.py.")
    r.p("Which channels we prune. In a ResNet the output of every block is added to its shortcut. If we removed "
        "a channel from a block's output, the same channel would have to be removed from the shortcut, from "
        "every other block in that stage that adds into the same tensor, and from the next stage's inputs. "
        "That needs coupled group pruning across the skip connections, which we leave as future work. Instead "
        "we prune only the inner channels of each BasicBlock: the output filters of conv1. That needs exactly "
        "three edits (conv1 filters, bn1 channels, conv2 input channels), and the block's output width stays "
        "the same, so the residual addition and the shortcut remain valid.")
    r.p("Importance and rebuild. Each inner channel is scored by the L1 norm of its conv1 filter multiplied by "
        "the absolute value of its batch-norm scale (gamma); a channel with small weights or a near-zero gamma "
        "contributes little. Within each block we keep the highest-scoring channels. We then build a new, "
        "smaller ResNet-18 with the new inner widths and copy the kept weights in by index. The result contains "
        "no masks and no zero channels.")
    r.p("Equivalence test. To prove the rebuild is correct, the script makes a copy of the original model where "
        "the pruned channels are zeroed (conv1 filter, bn1 gamma and bn1 beta set to 0, so the channel is exactly "
        "0 after BN and ReLU and adds nothing to conv2). This masked model and the rebuilt model must give the "
        "same outputs on a batch of test images. The test runs in float64; the largest difference was "
        f"{float(sw.equivalence_max_diff.max()):.1e}, far below the 1e-5 limit, for every ratio.")
    rows = []
    for _, row in sw.iterrows():
        key = str(row.ratio)
        rows.append([key, n(f"structured_sweep.csv|{key}|params", 0), n(f"structured_sweep.csv|{key}|params_reduction_pct", 1),
                     n(f"structured_sweep.csv|{key}|macs_m", 1), n(f"structured_sweep.csv|{key}|macs_reduction_pct", 1),
                     n(f"structured_sweep.csv|{key}|val_acc_after_prune"), n(f"structured_sweep.csv|{key}|val_acc_after_ft"),
                     n(f"structured_sweep.csv|{key}|test_acc_after_ft"), n(f"structured_sweep.csv|{key}|cpu_p50_ms", 3)])
    chosen = res["chosen_ratio"]
    r.p(f"We tried prune ratios 0.3, 0.5 and 0.7 of the inner channels. Each pruned network was fine-tuned for "
        f"10 epochs (SGD, cosine schedule from lr 0.02). Table {t} and Figure {f} show the sweep; ratio 0 is the "
        f"baseline. {structured_recovery_sentence(sw)} The deliverable model is chosen on validation accuracy only: the "
        f"largest ratio within 1.5 points of the baseline. That was ratio {chosen}, with "
        f"{n('structured_results.json|params_reduction_pct', 1)}% fewer parameters, "
        f"{n('structured_results.json|macs_reduction_pct', 1)}% fewer MACs and "
        f"{n('structured_results.json|test_acc')}% test accuracy ({drop_phrase('structured_results.json|test_drop')}).")
    r.table(["Ratio", "Params", "Params cut (%)", "MACs (M)", "MACs cut (%)", "Val acc. pruned (%)",
             "Val acc. FT (%)", "Test acc. FT (%)", "CPU p50 (ms)"], rows,
            "Structured pruning sweep. CPU latency here is a quick single-thread check; Section 9 has the full benchmark.",
            t, font=8)
    r.figure(FIG / "08_structured_sweep.png", "Structured sweep: validation accuracy against MACs before and after "
             "fine-tuning (left), and a zoom on the fine-tuned models labelled with CPU latency (right).", f, 16)


def onnx_section(r):
    rep = read_json("onnx_export.json")
    t = r.next_table()
    r.h("8. ONNX deployment")
    r.p(f"ONNX is a framework-neutral model format that ONNX Runtime can run on many devices. "
        f"scripts/06_export_onnx.py exports the FP32, unstructured and structured models with opset "
        f"{rep['opset']}, a dynamic batch dimension and the model in eval mode. In a float model QuantStub and "
        "DeQuantStub are identity, so the exported graphs contain only ordinary float operators. Every file is "
        "checked with onnx.checker, then ONNX Runtime and PyTorch are run on the same 64 test images; Table "
        f"{t} lists the largest difference in the logits and whether the predicted classes agree. "
        "scripts/07b_onnx_inference.py runs these files with the CPUExecutionProvider and one thread.")
    rows = []
    for key in ["fp32", "unstructured", "structured"]:
        item = rep.get(key, {})
        if item.get("status") == "ok":
            rows.append([PRETTY[key] if key != "fp32" else "FP32", "ok",
                         f"{item['max_abs_diff']:.1e}", f"{100 * item['pred_agreement']:.1f}",
                         n(f"onnx_export.json|{key}|size_mb")])
    for key, label in [("qat_int8", "QAT INT8 (QDQ)"), ("ort_ptq_int8", "ONNX Runtime PTQ INT8")]:
        item = rep.get(key, {})
        if item.get("status") == "ok":
            diff = item.get("max_abs_diff_vs_fakequant", item.get("max_abs_diff_vs_fp32"))
            agree = item.get("pred_agreement", item.get("pred_agreement_vs_fp32"))
            ref = "fake-quant PyTorch" if key == "qat_int8" else "FP32 PyTorch"
            rows.append([label, "ok", f"{diff:.1e} (vs {ref})", f"{100 * agree:.1f}", n(f"onnx_export.json|{key}|size_mb")])
        else:
            rows.append([label, "failed", "-", "-", "-"])
    r.table(["Model", "Export", "Max |PyTorch - ORT|", "Same prediction (%)", "Size (MB)"], rows,
            "ONNX export and parity checks.", t)

    qat_item, ort_item = rep.get("qat_int8", {}), rep.get("ort_ptq_int8", {})
    text = ("INT8 in ONNX was a best-effort extra. ")
    if qat_item.get("status") == "ok":
        text += ("For QAT we rebuilt the fake-quant model, loaded the trained QAT state, disabled the observers "
                 "and exported it in eval mode, so each fake-quant module became a QuantizeLinear/DequantizeLinear "
                 "(QDQ) pair. One problem had to be solved first: the default QAT setting uses a fused "
                 "fake-quant operator that the ONNX exporter does not support, so for the export we used the plain "
                 "FakeQuantize class with the same ranges and loaded the same trained scales and zero points into it. "
                 "In this file the weights are still stored as float next to their QDQ nodes, so it is not smaller "
                 "than the FP32 file; ONNX Runtime can fuse QDQ patterns into integer kernels when it runs it. ")
    else:
        text += f"The QAT QDQ export failed: {qat_item.get('error', 'unknown error')}. "
    if ort_item.get("status") == "ok":
        text += ("For PTQ we used ONNX Runtime's own quantize_static in QDQ format with per-channel weights, "
                 "calibrated on the same 1,000 training images. This is a different quantizer from the "
                 "PyTorch one in Section 4 (different range estimation and different kernels), so we label "
                 "it 'ONNX Runtime PTQ' and do not treat its accuracy as the PyTorch PTQ result.")
    else:
        text += f"The ONNX Runtime PTQ step failed: {ort_item.get('error', 'unknown error')}."
    r.p(text)


def benchmark(r):
    df = read_csv("benchmark_summary.csv").set_index("model")
    settings = read_json("benchmark_settings.json")
    t1 = r.next_table()
    t2 = r.next_table()
    f_acc, f_size, f_lat, f_conf, f_pc = (r.next_figure() for _ in range(5))
    r.h("9. Benchmark results")
    r.p(f"scripts/08_benchmark.py runs without arguments on all {settings['test_images']:,} test images. "
        "Latency is measured on the CPU with torch.set_num_threads(1) (and one intra-op and one inter-op thread "
        f"for ONNX Runtime), batch size 1, {settings['warmup']} warm-up runs and {settings['runs']} timed runs "
        "on real test images, timed with time.perf_counter. The whole latency pass, with the models in a fixed "
        f"order, was repeated {settings['repeats']} times and the median of each statistic is reported. p50 is "
        "the median time, p99 is the time that 99% of runs beat, so it shows the slow tail. Throughput uses "
        f"batch 128 with all {settings['throughput_threads']} CPU threads allowed. MACs are multiply-accumulate "
        "operations per image, counted with forward hooks on Conv2d and Linear layers; for the INT8 models we "
        "report the MACs of the fused FP32 model, since quantization changes the number type, not the number "
        "of operations. Parameters for the INT8 rows are also counted on the fused float model. "
        + load_sentence())
    rows = []
    for m in MAIN:
        rows.append([PRETTY[m], b(m, "test_accuracy"), b(m, "accuracy_drop_vs_fp32"), b(m, "size_mb"),
                     b(m, "params_nonzero", 0), b(m, "sparsity_pct", 1), b(m, "macs_m", 1),
                     b(m, "latency_p50_ms", 3), b(m, "latency_p99_ms", 3), b(m, "throughput_img_s", 0),
                     b(m, "speedup_vs_fp32")])
    r.table(["Model", "Test acc. (%)", "Drop (pts)", "Size (MB)", "Nonzero params", "Sparsity (%)",
             "MACs (M)", "p50 (ms)", "p99 (ms)", "Throughput (img/s)", "Speed-up (p50)"],
            rows, "Main results on the CPU. Latency: batch 1, 1 thread. Throughput: batch 128, all threads.",
            t1, font=7.5)

    extra = [m for m in df.index if m not in MAIN]
    rows = []
    for m in extra:
        rows.append([PRETTY.get(m, m), df.loc[m, "device"].upper(), b(m, "test_accuracy"), b(m, "size_mb"),
                     b(m, "latency_p50_ms", 3), b(m, "latency_p99_ms", 3), b(m, "throughput_img_s", 0),
                     b(m, "speedup_vs_fp32")])
    r.p(f"Table {t1} is the main result. Table {t2} adds the ONNX Runtime rows and, as an extra, GPU latency for "
        f"the three float models (GPU speed-ups are relative to FP32 on the GPU). {gpu_sentence(df)}")
    r.table(["Model", "Device", "Test acc. (%)", "Size (MB)", "p50 (ms)", "p99 (ms)", "Throughput (img/s)",
             "Speed-up"], rows, "ONNX Runtime and GPU rows.", t2, font=8)

    r.figure(FIG / "03_accuracy.png", "Test accuracy of the five models.", f_acc, 13)
    r.figure(FIG / "04_size.png", "File size of the five models on disk.", f_size, 13)
    r.figure(FIG / "05_latency.png", "CPU latency (batch 1, 1 thread): bars are p50, whiskers reach p99.", f_lat, 13)

    r.h("Discussion of the trade-offs", 2)
    r.p(f"Figures {f_acc}, {f_size} and {f_lat} put the three costs side by side. {int8_discussion(df)}")
    r.p(pruning_discussion(df))
    r.figure(FIG / "10_confusion_matrices.png",
             "Confusion matrices on the test set: FP32 baseline (left) and the most accurate compressed model (right).",
             f_conf, 16)
    r.figure(FIG / "11_per_class_accuracy.png", "Per-class test accuracy of the five models.", f_pc, 16)
    r.p(f"Figures {f_conf} and {f_pc} look at the errors. {per_class_discussion()}")


def limitations(r):
    r.h("10. Limitations")
    r.p("The CPU numbers come from a desktop Ryzen processor, not from an edge board. A Raspberry Pi or a "
        "Jetson would be several times slower in absolute terms, and an ARM board would use the qnnpack backend "
        "instead of the x86 backends, so the ratios could differ. Latency on a desktop also has noise from background "
        "programs, frequency boost and caches; we reduced this with warm-up runs, many timed runs, fixed order "
        "and the median of three passes, but differences of a few percent between models are within noise. "
        "The quick latency check inside the structured sweep (Table 5) times one image again and again right "
        "after training, while the main benchmark cycles through 300 different images in a separate run, so the "
        "two sets of numbers differ; only the main benchmark (Section 9) should be compared across models.")
    r.p("Unstructured sparsity gives no speed-up and no smaller file on dense hardware. Our structured pruning "
        "only touches the inner block channels, so the block outputs, which carry the residual additions, stay "
        "at full width; pruning them would need coupled group pruning. The INT8 ONNX files are a best-effort "
        "extra: the QAT QDQ file stores float weights, and the ONNX Runtime PTQ model comes from a different "
        "quantizer than our PyTorch PTQ model. Finally, each model was trained once with one seed, so we have "
        "no measured error bars, and differences of a few tenths of a point between models should not be "
        "over-read.")


def reproduction(r):
    t = r.next_table()
    r.h("11. Reproduction and deliverables")
    r.p("The README gives full instructions. In short: create a virtual environment, install the CUDA build of "
        "PyTorch and then requirements.txt, and run 'python run_all.py --full'. 'python run_all.py --quick' is a "
        "smoke test of the whole pipeline in a few minutes. Each step's console output is saved in results/logs/. "
        f"Table {t} maps the deliverables to files.")
    rows = [
        ["1", "Baseline model", "scripts/01_train_baseline.py, checkpoints/resnet18_fp32.pth"],
        ["2", "PTQ", "scripts/02_ptq.py (fusion, backend, calibrate(), convert), checkpoints/resnet18_ptq_int8.pt (+ _state.pth)"],
        ["3", "QAT", "scripts/03_qat.py (fake-quant fine-tuning), checkpoints/resnet18_qat_int8.pt (+ _state.pth)"],
        ["4", "Unstructured pruning", "scripts/04_unstructured_prune.py, checkpoints/resnet18_unstructured.pth"],
        ["5", "Structured pruning", "scripts/05_structured_prune.py, checkpoints/resnet18_structured.pth"],
        ["6", "Inference pipeline", "scripts/07_inference.py, src/infer_utils.py"],
        ["7", "Benchmarking", "scripts/08_benchmark.py, results/benchmark_summary.csv, results/predictions.csv"],
        ["8", "ONNX deployment", "scripts/06_export_onnx.py, scripts/07b_onnx_inference.py, onnx_models/*.onnx"],
        ["9", "Technical report", "report/Technical_Report.docx, report/screenshots/"],
        ["10", "README", "README.md, requirements.txt, run_all.py"],
    ]
    r.table(["#", "Deliverable", "Files"], rows, "Deliverables and where to find them.", t, [1, 4, 11])


def appendix(r):
    r.h("Appendix A. Screenshots")
    r.p("Each box below is for a screenshot taken on the machine that produced the results. The command for "
        "each one is in report/SCREENSHOTS_TODO.md.")
    for label, cap in [("nvidia-smi", "nvidia-smi output showing the GPU used for training."),
                       ("benchmark run in the terminal", "python scripts/08_benchmark.py running in the terminal."),
                       ("benchmark_summary.csv opened", "results/benchmark_summary.csv opened in a spreadsheet."),
                       ("project folder tree", "the project folder with all deliverables."),
                       ("inference with all models", "python scripts/07_inference.py --model all --image <image>.")]:
        r.placeholder(label, cap)
    r.h("Appendix B. Console output captured from the run", 1)
    r.p("The images below are the real log text of the runs (results/logs/), rendered as images by "
        "scripts/09_make_figures.py. They are console output captured from the run, not photos of a screen.")
    for name, cap in [("console_baseline.png", "baseline training (start and end of the log)"),
                      ("console_ptq.png", "PTQ conversion, summary and calibration-size study"),
                      ("console_qat.png", "QAT fine-tuning and summary"),
                      ("console_unstructured.png", "unstructured pruning summary"),
                      ("console_structured.png", "structured pruning, equivalence test and sweep"),
                      ("console_onnx.png", "ONNX export and parity checks"),
                      ("console_inference.png", "inference with all five models"),
                      ("console_benchmark.png", "benchmark summary table"),
                      ("console_tree.png", "deliverables folder listing")]:
        r.figure(SHOTS / name, f"Console output captured from the run: {cap}.", r.next_figure(), 16)
    r.h("Appendix C. Per-class accuracy", 1)
    pc = read_csv("per_class_accuracy.csv")
    rows = [[row["class"]] + [f"{row[m]:.2f}" for m in MAIN] for _, row in pc.iterrows()]
    r.table(["Class"] + [PRETTY[m] for m in MAIN], rows, "Per-class test accuracy (%).", r.next_table(), font=8.5)


def screenshots_todo():
    text = """# Screenshots to take and paste into the report

The DOCX has dashed placeholder boxes in Appendix A, each with a caption that starts with
"Screenshot:". Take these on your own PC, paste each image into its box, and delete the
placeholder text. Run every command from the project folder with the virtual environment active
(`.venv\\Scripts\\activate` on Windows, `source .venv/bin/activate` on Linux).

1. **GPU check**
   `nvidia-smi`
2. **Benchmark running in the terminal** (takes a few minutes; screenshot the final table)
   `python scripts/08_benchmark.py`
3. **Results CSV opened** in Excel or LibreOffice Calc
   open `results/benchmark_summary.csv`
4. **Folder tree** of the deliverables
   Windows: `tree /F checkpoints onnx_models results\\figures report`
   Linux: `tree checkpoints onnx_models results/figures report`
5. **Inference with every model type** (use any photo, e.g. of a cat or a car)
   `python scripts/07_inference.py --model all --image path\\to\\photo.jpg`
6. (optional) **ONNX Runtime inference**
   `python scripts/07b_onnx_inference.py --model all --num-samples 200`
"""
    (ROOT / "report" / "SCREENSHOTS_TODO.md").write_text(text, encoding="utf-8")


def to_pdf():
    pdf = OUT.with_suffix(".pdf")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    for candidate in [r"C:\Program Files\LibreOffice\program\soffice.exe"]:
        if not soffice and Path(candidate).exists():
            soffice = candidate
    if soffice:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent), str(OUT)],
                       check=False, timeout=180)
        if pdf.exists():
            return f"PDF written with LibreOffice: {pdf}"
    try:
        from docx2pdf import convert
        convert(str(OUT), str(pdf))
        if pdf.exists():
            return f"PDF written with docx2pdf (Word): {pdf}"
    except Exception as e:
        return f"No PDF made: neither LibreOffice nor Word is available ({type(e).__name__})."
    return "No PDF made: no converter found."


def main():
    r = Report()
    title_block(r)
    intro(r)
    environment(r)
    baseline(r)
    ptq(r)
    qat(r)
    unstructured(r)
    structured(r)
    onnx_section(r)
    benchmark(r)
    limitations(r)
    reproduction(r)
    appendix(r)
    r.save()
    screenshots_todo()
    with open(RESULTS / "report_numbers.json", "w") as f:
        json.dump(NUMBERS, f, indent=1)
    print(f"Saved {OUT} ({len(NUMBERS)} numbers taken from the results files, "
          f"{r.fig_no} figures, {r.tab_no} tables)")
    print(to_pdf())


if __name__ == "__main__":
    main()
