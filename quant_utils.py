"""Small helpers shared by the PTQ and QAT scripts: checking and saving INT8 models."""
from pathlib import Path

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def pick_backend():
    """Choose the INT8 CPU backend and make it the active quantized engine.

    On x86 desktops FBGEMM is PyTorch's usual INT8 backend, so we prefer it. The official Windows
    wheels of PyTorch 2.10/2.11 are built without FBGEMM (supported_engines is just ['onednn']),
    so there we fall back to oneDNN, Intel's x86 library, which also has INT8 conv/linear kernels
    and works on AMD CPUs. qnnpack would be the choice for ARM boards such as a Raspberry Pi.
    """
    engines = torch.backends.quantized.supported_engines
    for backend in ("fbgemm", "x86", "onednn"):
        if backend in engines:
            torch.backends.quantized.engine = backend
            return backend
    raise RuntimeError(f"no x86 INT8 backend in this PyTorch build: {engines}")


def use_backend(backend):
    """Activate one specific backend, e.g. the one an INT8 model was converted with."""
    engines = torch.backends.quantized.supported_engines
    if backend not in engines:
        raise RuntimeError(f"This INT8 model was converted with '{backend}', but this PyTorch build only "
                           f"supports {engines}. Re-run 02_ptq.py / 03_qat.py on this machine.")
    torch.backends.quantized.engine = backend


def check_all_convs_quantized(model):
    """Count quantized convs and make sure no plain float Conv2d is left after convert()."""
    float_convs = [n for n, m in model.named_modules() if type(m) is nn.Conv2d]
    int8_convs = [n for n, m in model.named_modules() if isinstance(m, nnq.Conv2d)]
    print(f"Quantized conv modules: {len(int8_convs)} | plain float Conv2d left: {len(float_convs)}")
    assert not float_convs, f"float convs left after convert: {float_convs}"
    return len(int8_convs)


def save_int8(model, name, example, config, extra):
    """Save the INT8 model as TorchScript (for deployment) and as a state_dict + config.

    The TorchScript file holds both the graph and the weights, so inference does not need
    our model class. Afterwards we reload it and check the predictions did not change.
    """
    CHECKPOINTS.mkdir(exist_ok=True)
    script_path = CHECKPOINTS / f"{name}.pt"
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(str(script_path))
    torch.save({"state_dict": model.state_dict(), "config": config,
                "backend": torch.backends.quantized.engine, **extra},
               CHECKPOINTS / f"{name}_state.pth")

    reloaded = torch.jit.load(str(script_path))
    with torch.no_grad():
        same = torch.equal(model(example).argmax(1), reloaded(example).argmax(1))
    print(f"TorchScript reload gives identical predictions on a test batch: {same}")
    assert same, "reloaded TorchScript model disagrees with the in-memory model"
    return script_path
