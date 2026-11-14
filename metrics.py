"""Model size, parameter counts, MACs and latency measurement."""
import gzip
import io
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero_params(model):
    return sum(int((p != 0).sum()) for p in model.parameters())


def prunable_layers(model):
    """All Conv2d and Linear layers (the ones we prune and count MACs for)."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def weight_sparsity(model):
    """Fraction of zeros over all Conv2d/Linear weights (not BN, not biases)."""
    total, zeros = 0, 0
    for _, m in prunable_layers(model):
        total += m.weight.numel()
        zeros += int((m.weight == 0).sum())
    return zeros / total


@torch.no_grad()
def count_macs(model, input_size=(1, 3, 32, 32)):
    """Multiply-accumulates for one image, counted with forward hooks on Conv2d and Linear.

    Conv: output pixels x out_channels x (in_channels / groups) x kh x kw.
    Linear: in_features x out_features. BN, ReLU, add and pooling are ignored (they are tiny).
    """
    macs = []

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        macs.append(out.shape[2] * out.shape[3] * m.out_channels * (m.in_channels // m.groups) * kh * kw)

    def linear_hook(m, inp, out):
        macs.append(m.in_features * m.out_features)

    hooks = []
    for _, m in prunable_layers(model):
        hook = conv_hook if isinstance(m, nn.Conv2d) else linear_hook
        hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(input_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return sum(macs)


def file_size_mb(path):
    return Path(path).stat().st_size / 1e6


def gzip_size_mb(path):
    """Size of the file after gzip (level 9). Shows what zeros from pruning save when stored."""
    buf = io.BytesIO()
    with open(path, "rb") as src, gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    return len(buf.getvalue()) / 1e6


def latency_stats(times_s):
    """Mean, std and percentiles in milliseconds from a list of times in seconds."""
    t = np.asarray(times_s) * 1000.0
    return {
        "latency_mean_ms": float(t.mean()),
        "latency_std_ms": float(t.std()),
        "latency_p50_ms": float(np.percentile(t, 50)),
        "latency_p95_ms": float(np.percentile(t, 95)),
        "latency_p99_ms": float(np.percentile(t, 99)),
    }


@torch.no_grad()
def measure_latency(run_fn, inputs, warmup=50, runs=300, sync_cuda=False):
    """Time run_fn(x) for single inputs. inputs is a list of ready tensors/arrays; reused in a cycle."""
    for i in range(warmup):
        run_fn(inputs[i % len(inputs)])
    if sync_cuda:
        torch.cuda.synchronize()
    times = []
    for i in range(runs):
        x = inputs[i % len(inputs)]
        start = time.perf_counter()
        run_fn(x)
        if sync_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return latency_stats(times)


def format_number(value, decimals):
    """How numbers are written in the report: fixed decimals, thousands separators for big integers.

    10_make_report.py and 11_verify_deliverables.py both use this, so their texts match exactly.
    """
    value = float(value)
    if decimals == 0 and abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.{decimals}f}"
