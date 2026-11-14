"""Train the FP32 ResNet-18 baseline on CIFAR-10 (deliverable 1).

SGD with Nesterov momentum, OneCycle learning rate (peak 0.1), weight decay 5e-4, batch 128,
bfloat16 mixed precision and channels_last on the GPU. The checkpoint with the best
validation accuracy is kept, and only that one is evaluated on the test set.

Run:  python scripts/01_train_baseline.py            (30 epochs)
      python scripts/01_train_baseline.py --quick    (1 epoch on a small subset)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_datasets, make_loader
from src.models import ResNet18
from src.train_utils import (CHECKPOINTS, RESULTS, History, Timer, current_lr, evaluate,
                             get_device, save_json, set_seed, train_one_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 1

    set_seed()
    device = get_device()
    print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    train_ds, val_ds, test_ds = get_datasets(quick=args.quick)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, 500, shuffle=False)
    test_loader = make_loader(test_ds, 500, shuffle=False)

    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    history = History("baseline")
    best_val, best_epoch = -1.0, -1
    ckpt_path = CHECKPOINTS / "resnet18_fp32.pth"
    CHECKPOINTS.mkdir(exist_ok=True)
    total = Timer()

    for epoch in range(1, args.epochs + 1):
        t = Timer()
        lr = current_lr(optimizer)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler,
                                                use_amp=True, channels_last=True,
                                                step_scheduler_per_batch=True)
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp=True, channels_last=True)
        history.log(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                    val_acc=val_acc, lr=lr, seconds=t.seconds())
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            # save in normal (contiguous) layout so the checkpoint loads anywhere
            state = {k: v.contiguous() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "config": model.config(), "val_acc": val_acc,
                        "epoch": epoch}, ckpt_path)

    # final test accuracy of the best-validation checkpoint, in plain FP32
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet18().to(device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, _ = evaluate(model, test_loader, device, use_amp=False)
    ckpt["test_acc"] = test_acc
    torch.save(ckpt, ckpt_path)

    summary = {"best_epoch": best_epoch, "best_val_acc": best_val, "test_acc": test_acc,
               "epochs": args.epochs, "train_minutes": total.seconds() / 60, "quick": args.quick}
    save_json(summary, RESULTS / "baseline_results.json")
    print(f"\nBest val acc {best_val:.2f}% at epoch {best_epoch}")
    print(f"Test accuracy (FP32 baseline): {test_acc:.2f}%")
    print(f"Training time: {total.seconds() / 60:.1f} min")
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
