"""Loading the five deployable models and running a prediction.

Used by scripts/07_inference.py and scripts/08_benchmark.py, so both load models the same way.
  fp32, unstructured, structured -> float checkpoints (.pth), rebuilt from the stored config
  ptq, qat                       -> TorchScript INT8 files (.pt), no class definition needed
INT8 models only have CPU kernels (FBGEMM / oneDNN), so they always run on CPU.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import CIFAR_MEAN, CIFAR_STD, CLASSES, eval_transform
from src.models import load_fp32_checkpoint
from src.quant_utils import use_backend

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"

MODEL_FILES = {
    "fp32": CKPT / "resnet18_fp32.pth",
    "ptq": CKPT / "resnet18_ptq_int8.pt",
    "qat": CKPT / "resnet18_qat_int8.pt",
    "unstructured": CKPT / "resnet18_unstructured.pth",
    "structured": CKPT / "resnet18_structured.pth",
}
MODEL_NAMES = list(MODEL_FILES)
INT8_MODELS = {"ptq", "qat"}


def int8_backend(name):
    """The quantized engine an INT8 model was converted with, as recorded next to its weights."""
    state = torch.load(CKPT / f"resnet18_{name}_int8_state.pth", map_location="cpu", weights_only=False)
    if "backend" in state:
        return state["backend"]
    # checkpoints made before the backend was stored: the PTQ/QAT result files record it
    with open(ROOT / "results" / f"{name}_results.json") as f:
        return json.load(f)["backend"]


def load_model(name, device="cpu"):
    """Return the model in eval mode on the right device. INT8 models are forced to CPU."""
    path = MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the script that creates the '{name}' model first.")
    if name in INT8_MODELS:
        use_backend(int8_backend(name))  # must be the backend the model was converted with
        model = torch.jit.load(str(path), map_location="cpu")
        device = "cpu"
    else:
        model, _ = load_fp32_checkpoint(path, map_location="cpu")
    model.eval()
    return model.to(device)


def preprocess_image(path):
    """Open any image, convert to RGB, resize to 32x32 and normalise like the test set."""
    img = Image.open(path).convert("RGB")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)
    return eval_transform()(img).unsqueeze(0)


@torch.no_grad()
def predict(model, x, device="cpu"):
    """Return (class index, confidence, latency in ms) for a single image tensor (1, 3, 32, 32)."""
    x = x.to(device)
    start = time.perf_counter()
    logits = model(x)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    probs = torch.softmax(logits.float(), dim=1)[0]
    idx = int(probs.argmax())
    return idx, float(probs[idx]), latency_ms


def denormalize(x):
    """Turn a normalised tensor back into a uint8 HWC image (used for saving sample images)."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


__all__ = ["CLASSES", "MODEL_NAMES", "MODEL_FILES", "INT8_MODELS", "load_model",
           "preprocess_image", "predict", "denormalize"]
