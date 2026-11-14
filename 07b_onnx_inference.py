"""CPU inference with ONNX Runtime (deliverable 8).

Uses CPUExecutionProvider with one intra-op and one inter-op thread, the same setting as the
latency benchmark.

Examples:
  python scripts/07b_onnx_inference.py --model fp32 --image path/to/img.png
  python scripts/07b_onnx_inference.py --model structured --num-samples 16
  python scripts/07b_onnx_inference.py --model all --num-samples 200   (accuracy check on test images)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort

from src.data import CLASSES, get_test_dataset
from src.infer_utils import preprocess_image

ONNX_DIR = Path(__file__).resolve().parents[1] / "onnx_models"
ONNX_FILES = {
    "fp32": ONNX_DIR / "resnet18_fp32.onnx",
    "unstructured": ONNX_DIR / "resnet18_unstructured.onnx",
    "structured": ONNX_DIR / "resnet18_structured.onnx",
    "qat_int8": ONNX_DIR / "resnet18_qat_int8_qdq.onnx",          # best effort
    "ort_ptq_int8": ONNX_DIR / "resnet18_ort_ptq_int8.onnx",      # best effort
}


def load_session(name):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(ONNX_FILES[name]), opts, providers=["CPUExecutionProvider"])


def predict(session, x):
    """x: numpy float32 array (1, 3, 32, 32). Returns (class index, confidence, latency ms)."""
    start = time.perf_counter()
    logits = session.run(None, {"input": x})[0][0]
    ms = (time.perf_counter() - start) * 1000
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(probs.argmax())
    return idx, float(probs[idx]), ms


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    parser.add_argument("--model", required=True, choices=list(ONNX_FILES) + ["all"])
    parser.add_argument("--image", type=str)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = [n for n in ONNX_FILES if ONNX_FILES[n].exists()] if args.model == "all" else [args.model]
    if args.image:
        x = preprocess_image(args.image).numpy()
        for name in names:
            session = load_session(name)
            predict(session, x)  # warm-up
            idx, conf, ms = predict(session, x)
            print(f"{name:13s} {CLASSES[idx]:11s} {100 * conf:6.2f}%  {ms:.2f} ms")
        return

    test = get_test_dataset()
    ids = np.random.RandomState(args.seed).choice(len(test), size=args.num_samples, replace=False)
    for name in names:
        session = load_session(name)
        correct, times = 0, []
        show = args.num_samples <= 20
        print(f"\nONNX model: {name} ({ONNX_FILES[name].name})")
        for i in ids:
            x, label = test[int(i)]
            idx, conf, ms = predict(session, x.unsqueeze(0).numpy())
            correct += int(idx == label)
            times.append(ms)
            if show:
                print(f"{i:6d}  true {CLASSES[label]:11s} pred {CLASSES[idx]:11s} {100 * conf:5.1f}%  {ms:.2f} ms")
        print(f"{correct}/{len(ids)} correct ({100 * correct / len(ids):.1f}%) | "
              f"median latency {np.median(times):.2f} ms")


if __name__ == "__main__":
    main()
