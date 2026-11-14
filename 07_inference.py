"""Inference with any of the five models (deliverable 6).

Examples:
  python scripts/07_inference.py --model fp32 --image path/to/img.png
  python scripts/07_inference.py --model ptq --num-samples 16        (random CIFAR-10 test images)
  python scripts/07_inference.py --model all --image path/to/img.png (compare all five)

Any image size works; it is resized to 32x32. INT8 models (ptq, qat) always run on the CPU.
Float models run on the CPU too unless --device cuda is given.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data import CLASSES, get_test_dataset
from src.infer_utils import INT8_MODELS, MODEL_NAMES, load_model, predict, preprocess_image


def run_on_image(names, image_path, device):
    x = preprocess_image(image_path)
    print(f"Image: {image_path}")
    print(f"{'model':13s} {'class':11s} {'confidence':>10s} {'latency ms':>10s}  device")
    for name in names:
        dev = "cpu" if name in INT8_MODELS else device
        model = load_model(name, dev)
        predict(model, x, dev)  # one warm-up call so the timing is not the first-call cost
        idx, conf, ms = predict(model, x, dev)
        print(f"{name:13s} {CLASSES[idx]:11s} {100 * conf:9.2f}% {ms:10.2f}  {dev}")


def run_on_samples(names, num_samples, device, seed):
    test = get_test_dataset()
    rng = np.random.RandomState(seed)
    ids = rng.choice(len(test), size=num_samples, replace=False)
    for name in names:
        dev = "cpu" if name in INT8_MODELS else device
        model = load_model(name, dev)
        predict(model, test[int(ids[0])][0].unsqueeze(0), dev)  # warm-up
        print(f"\nModel: {name} ({dev})")
        print(f"{'image_id':>8s}  {'true':11s} {'pred':11s} {'conf':>7s} {'ms':>6s}")
        correct, times = 0, []
        for i in ids:
            x, label = test[int(i)]
            idx, conf, ms = predict(model, x.unsqueeze(0), dev)
            correct += int(idx == label)
            times.append(ms)
            mark = "" if idx == label else "  <- wrong"
            print(f"{i:8d}  {CLASSES[label]:11s} {CLASSES[idx]:11s} {100 * conf:6.1f}% {ms:6.2f}{mark}")
        print(f"{correct}/{num_samples} correct | mean latency {np.mean(times):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Run a CIFAR-10 model on an image or on test samples.")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES + ["all"])
    parser.add_argument("--image", type=str, help="path to an image file (any size)")
    parser.add_argument("--num-samples", type=int, default=0, help="number of random test images")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    torch.set_num_threads(1)  # same setting as the latency benchmark
    names = MODEL_NAMES if args.model == "all" else [args.model]
    if args.image:
        run_on_image(names, args.image, args.device)
    else:
        run_on_samples(names, args.num_samples or 8, args.device, args.seed)


if __name__ == "__main__":
    main()
