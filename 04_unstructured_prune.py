"""Iterative global magnitude (L1) unstructured pruning with fine-tuning (deliverable 4).

All Conv2d and Linear weights are pruned together (global ranking), in 5 steps to cumulative
sparsity 40, 60, 75, 85 and 90 %. After each step the model is fine-tuned with the masks active.
After the last step there is a longer fine-tune at a lower learning rate, then the masks are made
permanent with prune.remove(), so the zeros live in the plain weight tensors.

Run:  python scripts/04_unstructured_prune.py
      python scripts/04_unstructured_prune.py --quick
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.utils.prune as prune

from src.data import get_datasets, make_loader
from src.metrics import (count_nonzero_params, count_params, file_size_mb, gzip_size_mb,
                         prunable_layers, weight_sparsity)
from src.models import load_fp32_checkpoint, save_gzip_checkpoint
from src.train_utils import (CHECKPOINTS, RESULTS, History, Timer, current_lr, evaluate,
                             get_device, save_json, set_seed, train_one_epoch)

TARGETS = [0.40, 0.60, 0.75, 0.85, 0.90]


def step_amount(current_sparsity, target_sparsity):
    """PyTorch prunes `amount` of the weights that are still unpruned, so convert the
    cumulative target into the fraction of the remaining weights to remove now."""
    return (target_sparsity - current_sparsity) / (1.0 - current_sparsity)


def fine_tune(model, train_loader, val_loader, device, epochs, lr, history, stage):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    val_acc = None
    for epoch in range(1, epochs + 1):
        t = Timer()
        lr_now = current_lr(optimizer)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device,
                                                scheduler, use_amp=True)
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp=True)
        history.log(stage=stage, epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc, lr=lr_now, seconds=t.seconds(),
                    sparsity_pct=100 * weight_sparsity(model))
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-per-step", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--final-lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs_per_step, args.final_epochs = 1, 1

    set_seed()
    device = get_device()
    model, ckpt = load_fp32_checkpoint(CHECKPOINTS / "resnet18_fp32.pth")
    model = model.to(device)
    train_ds, val_ds, test_ds = get_datasets(quick=args.quick)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, 500, shuffle=False)
    test_loader = make_loader(test_ds, 500, shuffle=False)

    layers = prunable_layers(model)
    params_to_prune = [(m, "weight") for _, m in layers]
    print(f"Pruning {len(params_to_prune)} layers (all Conv2d + Linear weights)")

    baseline_val, _ = evaluate(model, val_loader, device)
    print(f"Baseline val accuracy: {baseline_val:.2f}%")

    history = History("unstructured")
    iterations = []
    sparsity = 0.0
    total = Timer()
    for i, target in enumerate(TARGETS, start=1):
        amount = step_amount(sparsity, target)
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        sparsity = weight_sparsity(model)
        print(f"\nIteration {i}: target {100 * target:.0f}% | step amount {100 * amount:.1f}% of remaining "
              f"| measured sparsity {100 * sparsity:.2f}%")
        assert abs(sparsity - target) < 0.005, "measured sparsity is far from the target"

        acc_before, _ = evaluate(model, val_loader, device)
        acc_after = fine_tune(model, train_loader, val_loader, device, args.epochs_per_step,
                              args.lr, history, stage=f"iter{i}")
        iterations.append({"iteration": i, "target_sparsity_pct": 100 * target,
                           "step_amount_pct": round(100 * amount, 2),
                           "measured_sparsity_pct": round(100 * sparsity, 3),
                           "val_acc_before_ft": round(acc_before, 2),
                           "val_acc_after_ft": round(acc_after, 2)})

    print("\nFinal fine-tune at lower learning rate")
    final_val = fine_tune(model, train_loader, val_loader, device, args.final_epochs,
                          args.final_lr, history, stage="final")

    # make pruning permanent: weight = weight_orig * weight_mask, then drop the mask and the hook
    for module, name in params_to_prune:
        prune.remove(module, name)
    leftover = [n for n, _ in model.named_parameters() if n.endswith("_orig")]
    leftover += [n for n, _ in model.named_buffers() if n.endswith("_mask")]
    assert not leftover, f"pruning reparametrisation still present: {leftover[:3]}"
    final_sparsity = weight_sparsity(model)
    print(f"Masks removed. Sparsity is still {100 * final_sparsity:.2f}% in the plain weights.")

    test_acc, _ = evaluate(model, test_loader, device)
    model = model.cpu()
    path = CHECKPOINTS / "resnet18_unstructured.pth"
    torch.save({"state_dict": model.state_dict(), "config": model.config(), "val_acc": final_val,
                "test_acc": test_acc, "weight_sparsity": final_sparsity}, path)
    # the deployable compressed file: same weights, gzip-compressed (most of them are zero)
    gz_path = CHECKPOINTS / "resnet18_unstructured.pth.gz"
    save_gzip_checkpoint(path, gz_path)

    # per-layer sparsity table
    layer_rows = []
    for name, m in prunable_layers(model):
        n = m.weight.numel()
        z = int((m.weight == 0).sum())
        layer_rows.append({"layer": name, "type": type(m).__name__, "weights": n, "zeros": z,
                           "sparsity_pct": round(100 * z / n, 2)})
    with open(RESULTS / "unstructured_layer_sparsity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    iterations.append({"iteration": "final", "target_sparsity_pct": 90.0, "step_amount_pct": 0.0,
                       "measured_sparsity_pct": round(100 * final_sparsity, 3),
                       "val_acc_before_ft": iterations[-1]["val_acc_after_ft"],
                       "val_acc_after_ft": round(final_val, 2)})
    with open(RESULTS / "unstructured_iterations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(iterations[0]))
        writer.writeheader()
        writer.writerows(iterations)

    fp32_path = CHECKPOINTS / "resnet18_fp32.pth"
    summary = {
        "baseline_val_acc": baseline_val, "baseline_test_acc": ckpt["test_acc"],
        "final_val_acc": final_val, "test_acc": test_acc,
        "test_drop": ckpt["test_acc"] - test_acc,
        "weight_sparsity_pct": 100 * final_sparsity,
        "params_total": count_params(model), "params_nonzero": count_nonzero_params(model),
        "size_mb": file_size_mb(path), "gzip_size_mb": file_size_mb(gz_path),
        "fp32_size_mb": file_size_mb(fp32_path), "fp32_gzip_size_mb": gzip_size_mb(fp32_path),
        "train_minutes": total.seconds() / 60, "quick": args.quick,
    }
    save_json(summary, RESULTS / "unstructured_results.json")

    print("\n===== Unstructured pruning summary =====")
    for row in iterations:
        print(f"iter {row['iteration']!s:>5}: sparsity {row['measured_sparsity_pct']:6.2f}% | "
              f"val before FT {row['val_acc_before_ft']:6.2f}% | after FT {row['val_acc_after_ft']:6.2f}%")
    print(f"Test accuracy: {test_acc:.2f}% (baseline {ckpt['test_acc']:.2f}%)")
    print(f"Nonzero params: {summary['params_nonzero']:,} of {summary['params_total']:,}")
    print(f"File size {summary['size_mb']:.2f} MB (dense, same as baseline) | "
          f"gzip {summary['gzip_size_mb']:.2f} MB vs baseline gzip {summary['fp32_gzip_size_mb']:.2f} MB")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
