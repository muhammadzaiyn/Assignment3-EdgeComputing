"""Structured channel pruning with a rebuilt, genuinely smaller ResNet-18 (deliverable 5).

Inside each of the 8 BasicBlocks we remove inner channels: output filters of conv1. That needs
exactly three connected edits: drop those conv1 filters, drop the matching bn1 channels, and drop
the matching input channels of conv2. The block's output width does not change, so the residual
add and the shortcut stay valid. Channels that feed residual adds are left alone on purpose
(they must match across the skip connection, see the report).

For each prune ratio in {0.3, 0.5, 0.7}: score channels, rebuild a smaller network, prove the
rebuild is exact with an equivalence test, fine-tune, and measure size / MACs / CPU latency.
The deliverable is the largest ratio within 1.5 points of the baseline on validation.

Run:  python scripts/05_structured_prune.py
      python scripts/05_structured_prune.py --quick
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import get_datasets, make_loader
from src.metrics import count_macs, count_params, file_size_mb, measure_latency
from src.models import ResNet18, load_fp32_checkpoint
from src.train_utils import (CHECKPOINTS, RESULTS, History, Timer, current_lr, evaluate,
                             get_device, save_json, set_seed, train_one_epoch)

RATIOS = [0.3, 0.5, 0.7]
MAX_VAL_DROP = 1.5  # accuracy points


def channel_scores(block):
    """Importance of each inner channel: L1 norm of its conv1 filter times |BN gamma|.

    A filter with small weights, or whose BN scale is close to zero, contributes little.
    """
    l1 = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    gamma = block.bn1.weight.detach().abs()
    return l1 * gamma


def choose_kept_channels(model, ratio):
    """For every block, the sorted indices of the inner channels to keep."""
    kept = []
    for block in model.blocks():
        scores = channel_scores(block)
        n_keep = max(1, int(round(len(scores) * (1 - ratio))))
        keep = torch.argsort(scores, descending=True)[:n_keep]
        kept.append(torch.sort(keep).values)
    return kept


def rebuild_pruned(model, kept):
    """Build a new, smaller ResNet18 and copy the kept weights by index. No masks, no zeros."""
    small = ResNet18(num_classes=model.num_classes, mid_channels=[len(k) for k in kept])
    big_state = model.state_dict()
    new_state = small.state_dict()

    for name in new_state:
        new_state[name] = big_state[name].clone()  # most tensors are unchanged

    for b, keep in enumerate(kept):
        prefix = block_prefix(b)
        new_state[f"{prefix}.conv1.weight"] = big_state[f"{prefix}.conv1.weight"][keep].clone()
        for p in ["weight", "bias", "running_mean", "running_var"]:
            new_state[f"{prefix}.bn1.{p}"] = big_state[f"{prefix}.bn1.{p}"][keep].clone()
        new_state[f"{prefix}.conv2.weight"] = big_state[f"{prefix}.conv2.weight"][:, keep].clone()

    small.load_state_dict(new_state)
    return small


def block_prefix(b):
    """Name of block b (0..7) in the state dict, e.g. 'layer2.1'."""
    return f"layer{b // 2 + 1}.{b % 2}"


@torch.no_grad()
def equivalence_test(model, small, kept, images):
    """The masked original (pruned channels zeroed) must give the same outputs as the rebuild.

    Zeroing the conv1 filter, BN gamma and BN beta makes the channel exactly 0 after BN and ReLU,
    so it adds nothing to conv2. If the copy-by-index is right, both models agree. We run the test
    in float64 so the only possible difference is a real bug, not rounding.
    """
    masked = copy.deepcopy(model).double().eval().cpu()
    for block, keep in zip(masked.blocks(), kept):
        drop = torch.ones(block.conv1.out_channels, dtype=torch.bool)
        drop[keep] = False
        block.conv1.weight[drop] = 0
        block.bn1.weight[drop] = 0
        block.bn1.bias[drop] = 0
    small64 = copy.deepcopy(small).double().eval().cpu()
    x = images.double()
    max_diff = (masked(x) - small64(x)).abs().max().item()
    assert max_diff < 1e-5, f"rebuilt model differs from masked model by {max_diff}"
    return max_diff


def cpu_latency_ms(model, image, runs=200):
    """Quick single-thread CPU p50 latency for the sweep table (the full benchmark is in 08)."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = copy.deepcopy(model).cpu().eval()
    stats = measure_latency(model, [image], warmup=30, runs=runs)
    torch.set_num_threads(old_threads)
    return stats["latency_p50_ms"]


def fine_tune(model, train_loader, val_loader, device, epochs, lr, history, ratio):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        t = Timer()
        lr_now = current_lr(optimizer)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler,
                                                use_amp=True)
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp=True)
        history.log(ratio=ratio, epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc, lr=lr_now, seconds=t.seconds())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 1

    set_seed()
    device = get_device()
    base, ckpt = load_fp32_checkpoint(CHECKPOINTS / "resnet18_fp32.pth")
    base.eval()
    train_ds, val_ds, test_ds = get_datasets(quick=args.quick)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, 500, shuffle=False)
    test_loader = make_loader(test_ds, 500, shuffle=False)
    test_images = next(iter(test_loader))[0][:32]
    one_image = test_images[:1]

    base_val, _ = evaluate(base.to(device), val_loader, device)
    base_params = count_params(base)
    base_macs = count_macs(base)
    base_latency = cpu_latency_ms(base, one_image)
    print(f"Baseline: val {base_val:.2f}% | params {base_params:,} | MACs {base_macs / 1e6:.1f} M "
          f"| CPU p50 {base_latency:.2f} ms")

    history = History("structured")
    sweep = [{"ratio": 0.0, "mid_channels": " ".join(map(str, base.mid_channels)),
              "params": base_params, "params_reduction_pct": 0.0, "macs_m": round(base_macs / 1e6, 2),
              "macs_reduction_pct": 0.0, "equivalence_max_diff": 0.0,
              "val_acc_after_prune": round(base_val, 2), "val_acc_after_ft": round(base_val, 2),
              "test_acc_after_ft": round(ckpt["test_acc"], 2), "cpu_p50_ms": round(base_latency, 3)}]
    pruned_models = {}
    (CHECKPOINTS / "intermediate").mkdir(parents=True, exist_ok=True)

    for ratio in RATIOS:
        print(f"\n--- prune ratio {ratio} of the inner channels ---")
        base_cpu = base.cpu()
        kept = choose_kept_channels(base_cpu, ratio)
        small = rebuild_pruned(base_cpu, kept)
        max_diff = equivalence_test(base_cpu, small, kept, test_images)
        print(f"New inner widths: {small.mid_channels}")
        print(f"Equivalence test passed: max |masked - rebuilt| = {max_diff:.2e} (float64)")

        small = small.to(device)
        val_pruned, _ = evaluate(small, val_loader, device)
        print(f"Val accuracy right after pruning: {val_pruned:.2f}%")
        val_ft = fine_tune(small, train_loader, val_loader, device, args.epochs, args.lr, history, ratio)
        test_ft, _ = evaluate(small, test_loader, device)
        small = small.cpu()
        params = count_params(small)
        macs = count_macs(small)
        latency = cpu_latency_ms(small, one_image)
        print(f"After fine-tuning: val {val_ft:.2f}% | test {test_ft:.2f}% | params {params:,} "
              f"| MACs {macs / 1e6:.1f} M | CPU p50 {latency:.2f} ms")

        sweep.append({"ratio": ratio, "mid_channels": " ".join(map(str, small.mid_channels)),
                      "params": params, "params_reduction_pct": round(100 * (1 - params / base_params), 2),
                      "macs_m": round(macs / 1e6, 2),
                      "macs_reduction_pct": round(100 * (1 - macs / base_macs), 2),
                      "equivalence_max_diff": max_diff,
                      "val_acc_after_prune": round(val_pruned, 2), "val_acc_after_ft": round(val_ft, 2),
                      "test_acc_after_ft": round(test_ft, 2), "cpu_p50_ms": round(latency, 3)})
        pruned_models[ratio] = (small, val_ft, test_ft)
        torch.save({"state_dict": small.state_dict(), "config": small.config(), "val_acc": val_ft,
                    "test_acc": test_ft, "ratio": ratio},
                   CHECKPOINTS / "intermediate" / f"structured_ratio{ratio}.pth")

    with open(RESULTS / "structured_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    # choose on validation only: the largest ratio within MAX_VAL_DROP of the baseline
    ok = [r for r in RATIOS if base_val - pruned_models[r][1] <= MAX_VAL_DROP]
    chosen = max(ok) if ok else min(RATIOS)
    if not ok:
        print(f"No ratio stayed within {MAX_VAL_DROP} points; falling back to the smallest ratio.")
    small, val_ft, test_ft = pruned_models[chosen]
    path = CHECKPOINTS / "resnet18_structured.pth"
    torch.save({"state_dict": small.state_dict(), "config": small.config(), "val_acc": val_ft,
                "test_acc": test_ft, "ratio": chosen}, path)

    chosen_row = next(r for r in sweep if r["ratio"] == chosen)
    summary = {"chosen_ratio": chosen, "rule": f"largest ratio with val drop <= {MAX_VAL_DROP} points",
               "baseline_val_acc": base_val, "baseline_test_acc": ckpt["test_acc"],
               "val_acc": val_ft, "test_acc": test_ft, "test_drop": ckpt["test_acc"] - test_ft,
               "params": chosen_row["params"], "params_reduction_pct": chosen_row["params_reduction_pct"],
               "macs_m": chosen_row["macs_m"], "macs_reduction_pct": chosen_row["macs_reduction_pct"],
               "mid_channels": small.mid_channels, "size_mb": file_size_mb(path),
               "quick": args.quick}
    save_json(summary, RESULTS / "structured_results.json")

    print("\n===== Structured pruning summary =====")
    print(f"{'ratio':>5} {'params':>11} {'MACs(M)':>8} {'val pruned':>10} {'val FT':>7} {'test FT':>7} {'CPU p50 ms':>10}")
    for r in sweep:
        print(f"{r['ratio']:>5} {r['params']:>11,} {r['macs_m']:>8} {r['val_acc_after_prune']:>10} "
              f"{r['val_acc_after_ft']:>7} {r['test_acc_after_ft']:>7} {r['cpu_p50_ms']:>10}")
    print(f"Chosen ratio {chosen}: test {test_ft:.2f}% (baseline {ckpt['test_acc']:.2f}%), "
          f"{chosen_row['params_reduction_pct']}% fewer params, {chosen_row['macs_reduction_pct']}% fewer MACs")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
