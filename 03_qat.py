"""Quantization-aware training (QAT) starting from the FP32 baseline (deliverable 3).

Steps: load FP32 weights -> fuse for QAT (train mode) -> qat qconfig -> prepare_qat (inserts
fake-quant modules) -> fine-tune on the GPU without AMP -> freeze BN statistics and observers
part-way through -> move to CPU -> convert to INT8 -> evaluate the real INT8 model -> save.

Run:  python scripts/03_qat.py            (8 epochs)
      python scripts/03_qat.py --quick
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.ao.nn.intrinsic.qat as nniqat
from torch.ao.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat

from src.data import get_datasets, make_loader
from src.metrics import file_size_mb
from src.models import load_fp32_checkpoint
from src.quant_utils import check_all_convs_quantized, pick_backend, save_int8
from src.train_utils import (CHECKPOINTS, RESULTS, History, Timer, current_lr, evaluate,
                             get_device, save_json, set_seed, train_one_epoch)


def prepare_qat_model(fp32_model):
    """Fuse in train mode and insert fake-quant modules. Returns the QAT-ready model."""
    model = copy.deepcopy(fp32_model).train()
    backend = pick_backend()  # same INT8 backend as PTQ
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-bn-epoch", type=int, default=3,
                        help="BN running stats are frozen from this epoch on (1-based)")
    parser.add_argument("--freeze-observer-epoch", type=int, default=4,
                        help="observers are disabled from this epoch on (1-based)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs, args.freeze_bn_epoch, args.freeze_observer_epoch = 2, 2, 2

    set_seed()
    device = get_device()
    cpu = torch.device("cpu")

    fp32_model, ckpt = load_fp32_checkpoint(CHECKPOINTS / "resnet18_fp32.pth")
    train_ds, val_ds, test_ds = get_datasets(quick=args.quick)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, 500, shuffle=False)
    test_loader = make_loader(test_ds, 250, shuffle=False, num_workers=2)

    qat_model = prepare_qat_model(fp32_model).to(device)
    n_fake = sum(1 for m in qat_model.modules() if type(m).__name__ == "FusedMovingAvgObsFakeQuantize"
                 or type(m).__name__ == "FakeQuantize")
    print(f"Fake-quant modules inserted: {n_fake}")

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = History("qat")
    total = Timer()
    for epoch in range(1, args.epochs + 1):
        # BN freeze: from here BN uses its running statistics, matching what convert() folds in
        if epoch == args.freeze_bn_epoch:
            qat_model.apply(nniqat.freeze_bn_stats)
            print(f"Epoch {epoch}: BN statistics frozen")
        # observer freeze: quantization ranges (scale / zero point) stop moving
        if epoch == args.freeze_observer_epoch:
            qat_model.apply(disable_observer)
            print(f"Epoch {epoch}: observers disabled")

        t = Timer()
        lr = current_lr(optimizer)
        # no AMP here: bfloat16 would interfere with the observers' ranges
        train_loss, train_acc = train_one_epoch(qat_model, train_loader, optimizer, device,
                                                scheduler, use_amp=False)
        val_acc, val_loss = evaluate(qat_model, val_loader, device)
        history.log(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                    val_acc=val_acc, lr=lr, seconds=t.seconds(),
                    bn_frozen=int(epoch >= args.freeze_bn_epoch),
                    observers_off=int(epoch >= args.freeze_observer_epoch))

    # keep the trained fake-quant model so the ONNX export can produce a QDQ graph from it
    (CHECKPOINTS / "intermediate").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": qat_model.state_dict(), "config": ckpt["config"]},
               CHECKPOINTS / "intermediate" / "qat_prepared.pth")
    fake_quant_val = val_acc

    # convert the real INT8 model on the CPU and evaluate it there
    qat_model = qat_model.to(cpu).eval()
    int8_model = convert(qat_model)
    n_qconv = check_all_convs_quantized(int8_model)
    val_acc, _ = evaluate(int8_model, val_loader, cpu)
    test_acc, _ = evaluate(int8_model, test_loader, cpu)

    example = next(iter(test_loader))[0][:16]
    script_path = save_int8(int8_model, "resnet18_qat_int8", example, ckpt["config"],
                                {"val_acc": val_acc, "test_acc": test_acc})

    fp32_size = file_size_mb(CHECKPOINTS / "resnet18_fp32.pth")
    int8_size = file_size_mb(script_path)
    fp32_test = ckpt["test_acc"]
    summary = {
        "backend": pick_backend(), "epochs": args.epochs, "lr": args.lr,
        "freeze_bn_epoch": args.freeze_bn_epoch, "freeze_observer_epoch": args.freeze_observer_epoch,
        "fake_quant_modules": n_fake, "quantized_conv_modules": n_qconv,
        "fake_quant_val_acc": fake_quant_val, "qat_val_acc": val_acc, "qat_test_acc": test_acc,
        "fp32_test_acc": fp32_test, "test_drop": fp32_test - test_acc,
        "fp32_size_mb": fp32_size, "qat_size_mb": int8_size, "size_ratio": fp32_size / int8_size,
        "train_minutes": total.seconds() / 60, "quick": args.quick,
    }
    save_json(summary, RESULTS / "qat_results.json")

    print("\n===== QAT summary =====")
    print(f"FP32 test accuracy      : {fp32_test:.2f}%")
    print(f"QAT INT8 test accuracy  : {test_acc:.2f}%  (drop {fp32_test - test_acc:.2f} points)")
    print(f"QAT INT8 val accuracy   : {val_acc:.2f}%  (fake-quant on GPU: {fake_quant_val:.2f}%)")
    print(f"File size FP32 / INT8   : {fp32_size:.2f} MB / {int8_size:.2f} MB  ({fp32_size / int8_size:.2f}x smaller)")
    print(f"Fine-tuning time        : {total.seconds() / 60:.1f} min")
    print(f"Saved {script_path}")


if __name__ == "__main__":
    main()
