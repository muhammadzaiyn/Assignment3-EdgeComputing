# Quantization and Pruning of ResNet-18 on CIFAR-10

Edge Computing & Intelligent Systems assignment.
Muhammad Zain ul Abidin, 23I-0051, AI-A.

We train one ResNet-18 on CIFAR-10 and turn it into four smaller or faster versions:

| # | Model | How it is made | Script |
|---|-------|----------------|--------|
| 1 | FP32 baseline | normal training on the GPU | `scripts/01_train_baseline.py` |
| 2 | PTQ INT8 | post-training static quantization (fuse, calibrate, convert) | `scripts/02_ptq.py` |
| 3 | QAT INT8 | quantization-aware fine-tuning, then convert | `scripts/03_qat.py` |
| 4 | Unstructured-pruned | iterative global L1 magnitude pruning to 90 % sparsity | `scripts/04_unstructured_prune.py` |
| 5 | Structured-pruned | inner block channels removed, network rebuilt smaller | `scripts/05_structured_prune.py` |

Then the models are exported to ONNX, run with one inference script, and compared by one
benchmark script on the CPU. The technical report is built from the result files.

## Folder map

```
README.md                  this file
requirements.txt           exact package versions used (install PyTorch first, see below)
run_all.py                 runs every step in order (--quick or --full)
src/
  data.py                  CIFAR-10 download, 45k/5k/10k split, loaders, calibration subset
  models.py                CIFAR ResNet-18, quantization-friendly, per-block inner widths
  train_utils.py           training loop, evaluation, seeding, history logging
  metrics.py               parameter / nonzero counts, MACs, file size, latency statistics
  quant_utils.py           INT8 backend choice, checks and saving for PTQ/QAT
  infer_utils.py           load_model(name) for all five models, predict()
scripts/
  01_train_baseline.py     baseline training                       (deliverable 1)
  02_ptq.py                PTQ: fusion, backend, calibration, convert (deliverable 2)
  03_qat.py                QAT fine-tuning and convert              (deliverable 3)
  04_unstructured_prune.py unstructured pruning + fine-tuning       (deliverable 4)
  05_structured_prune.py   structured channel pruning + rebuild     (deliverable 5)
  06_export_onnx.py        ONNX export and parity checks            (deliverable 8)
  07_inference.py          inference CLI for all five models        (deliverable 6)
  07b_onnx_inference.py    ONNX Runtime CPU inference               (deliverable 8)
  08_benchmark.py          automatic benchmark -> CSV               (deliverable 7)
  09_make_figures.py       figures and console-capture images
  10_make_report.py        builds report/Technical_Report.docx       (deliverable 9)
  11_verify_deliverables.py  automatic self-check (PASS/FAIL)
  12_package.py            builds the submission ZIP
checkpoints/               the trained models
onnx_models/               exported ONNX models
results/                   CSV / JSON results, figures/, logs/
report/                    Technical_Report.docx, screenshots/, SCREENSHOTS_TODO.md
docs/VIVA_NOTES.md         what each file does, and likely viva questions
```

## Requirements

* Python 3.14 (what the results were made with, on Windows 11). The pinned versions in
  requirements.txt need a recent Python; with an older Python (3.11 or later) install the newest
  versions of the same packages instead of the pinned ones.
* An NVIDIA GPU with a recent driver for training (tested on an RTX 5060 Ti 16 GB). The benchmark
  itself runs on the CPU.
* About 5 GB of free disk space (PyTorch with CUDA is about 3 GB, CIFAR-10 about 180 MB).

## Installation

### Windows (PowerShell)

```
cd edge_assignment
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Linux (bash)

```
cd edge_assignment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Pick the `cuXXX` index that matches your driver (`nvidia-smi` shows the highest CUDA version it
supports). RTX 50-series (Blackwell) cards need cu128 or newer. The last line must print `True`.

## Running

Everything is run from the project root.

Smoke test of the whole pipeline (tiny runs, about 11 minutes; it overwrites the results, so run
the full pipeline afterwards):

```
python run_all.py --quick
```

Full run (about 1 hour on an RTX 5060 Ti; console output of each step goes to `results/logs/`):

```
python run_all.py --full
```

Resume from a step after fixing something, for example from the unstructured pruning:

```
python run_all.py --full --from 04
```

Or run the steps one by one:

```
python scripts/01_train_baseline.py        # 30 epochs, saves checkpoints/resnet18_fp32.pth
python scripts/02_ptq.py                   # INT8 PTQ model + calibration-size study
python scripts/03_qat.py                   # 8 epochs of QAT, INT8 QAT model
python scripts/04_unstructured_prune.py    # 5 pruning steps to 90 % sparsity + fine-tuning
python scripts/05_structured_prune.py      # sweep of ratios 0.3 / 0.5 / 0.7, picks one
python scripts/06_export_onnx.py           # ONNX files + parity checks
python scripts/08_benchmark.py             # all CSVs in results/
python scripts/09_make_figures.py
python scripts/10_make_report.py
python scripts/12_package.py               # builds 23I-0051_AI-A.zip
python scripts/11_verify_deliverables.py   # PASS/FAIL for every check
```

Every training script accepts `--quick` and has options such as `--epochs` and `--lr`
(see `--help`).

### Inference

```
python scripts/07_inference.py --model fp32 --image path/to/photo.jpg
python scripts/07_inference.py --model ptq --num-samples 16
python scripts/07_inference.py --model all --image path/to/photo.jpg
python scripts/07b_onnx_inference.py --model structured --num-samples 16
python scripts/07b_onnx_inference.py --model all --num-samples 200
```

`--model` is one of `fp32, ptq, qat, unstructured, structured, all`. Any image size works (it is
resized to 32x32). INT8 models always run on the CPU.

## Expected run time (RTX 5060 Ti, Ryzen 7 7700)

| Step | Time |
|------|------|
| Dataset download | 1-25 min (depends on the server) |
| Baseline, 30 epochs | about 17 min |
| PTQ + calibration study | about 7 min (mostly CPU evaluation) |
| QAT, 8 epochs | about 7 min |
| Unstructured pruning, 30 epochs in total | about 7 min |
| Structured sweep, 3 x 10 epochs | about 7 min |
| ONNX, benchmark, figures, report, checks | about 17 min |

## Where the outputs go

* `checkpoints/` the five models (`resnet18_fp32.pth`, `resnet18_ptq_int8.pt`, `resnet18_qat_int8.pt`,
  `resnet18_unstructured.pth`, `resnet18_structured.pth`) plus the INT8 state dicts and
  `resnet18_unstructured.pth.gz`, the 90 %-sparse model gzip-compressed (7 MB instead of 45 MB;
  `load_fp32_checkpoint` in `src/models.py` reads it directly).
* `onnx_models/` the ONNX files.
* `results/benchmark_summary.csv` the main table, `results/predictions.csv` one row per test image,
  plus per-class accuracy, pruning tables, histories, `system_info.json` and `figures/`.
* `report/Technical_Report.docx` the report.

## Troubleshooting

**`torch.cuda.is_available()` is False.** You installed the CPU build. Uninstall torch and
torchvision and install them again from the CUDA index above. Check that `nvidia-smi` works.
A message like "no kernel image is available" means the CUDA build is too old for the GPU;
use cu128 or newer for RTX 50-series cards.

**The dataset download is very slow.** The official server is sometimes slow. `src/data.py`
checks the speed after 3 minutes and switches to the Hugging Face copy (`uoft-cs/cifar10`) if it gets
less than 3 MB per minute or would need more than 30 more minutes. The data is cached in `data/` after the first run.

**Exporting the QAT model to ONNX from the ZIP.** The QDQ export in `06_export_onnx.py` needs the
trained fake-quant model `checkpoints/intermediate/qat_prepared.pth`, which `03_qat.py` writes but
the ZIP leaves out (it is an intermediate file). Run `03_qat.py` first; the other ONNX exports work
without it, and a failure of this best-effort step is recorded instead of stopping the pipeline.

**"quantized engine FBGEMM is not supported".** Some PyTorch builds (for example the official
Windows wheels of 2.10 and 2.11) do not include FBGEMM. `src/quant_utils.pick_backend()` then uses
oneDNN, which is also an x86 INT8 backend. The backend is stored with each INT8 model and
`load_model` switches to exactly that one; if this PyTorch build lacks it, you get a clear error
and should re-run `02_ptq.py` and `03_qat.py` on that machine.

**DataLoader errors on Windows.** All entry points use `if __name__ == "__main__":`. If you still
see worker errors, lower `num_workers` in `src/data.make_loader`.
