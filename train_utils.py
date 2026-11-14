"""Training loop, evaluation, seeding and small logging helpers shared by all scripts."""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Training scripts need an NVIDIA GPU.")
    return torch.device("cuda")


def train_one_epoch(model, loader, optimizer, device, scheduler=None, use_amp=True,
                    channels_last=False, step_scheduler_per_batch=False):
    """One pass over the training data. Returns (mean loss, accuracy %)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)

        # bfloat16 autocast: same speed as fp16 on recent GPUs and no loss scaler needed
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None and step_scheduler_per_batch:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    if scheduler is not None and not step_scheduler_per_batch:
        scheduler.step()
    return total_loss / seen, 100.0 * correct / seen


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, channels_last=False):
    """Accuracy % and mean loss on a loader. Works for float and INT8 (CPU) models."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        total_loss += criterion(logits.float(), labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return 100.0 * correct / seen, total_loss / seen


@torch.no_grad()
def predict_all(model, loader, device):
    """Return (predicted labels, confidence of the prediction, true labels) as numpy arrays."""
    model.eval()
    preds, confs, trues = [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)).float(), dim=1)
        conf, pred = probs.max(1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
        trues.append(labels)
    return torch.cat(preds).numpy(), torch.cat(confs).numpy(), torch.cat(trues).numpy()


class History:
    """Collects one row per epoch, prints it, and writes results/<name>_history.csv."""

    def __init__(self, name):
        self.path = RESULTS / f"{name}_history.csv"
        self.rows = []

    def log(self, **row):
        self.rows.append(row)
        parts = []
        for key, value in row.items():
            if key == "lr":
                parts.append(f"lr {value:.5f}")
            elif isinstance(value, float):
                parts.append(f"{key} {value:.2f}")
            else:
                parts.append(f"{key} {value}")
        print(" | ".join(parts), flush=True)
        self.save()

    def save(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


class Timer:
    def __init__(self):
        self.start = time.time()

    def seconds(self):
        return time.time() - self.start


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
