"""Run the whole pipeline in order, saving each step's console output to results/logs/.

  python run_all.py --quick          smoke test: tiny runs that check the plumbing (a few minutes)
  python run_all.py --full           the real run (about 1.5-2 hours on an RTX-class GPU)
  python run_all.py --full --from 04 resume from a step (here: unstructured pruning)

Steps 01-08 create models and results, 09-10 build figures and the report,
12 builds the submission ZIP and 11 (run last) checks every deliverable, including the ZIP.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"

STEPS = [
    ("01", "scripts/01_train_baseline.py"),
    ("02", "scripts/02_ptq.py"),
    ("03", "scripts/03_qat.py"),
    ("04", "scripts/04_unstructured_prune.py"),
    ("05", "scripts/05_structured_prune.py"),
    ("06", "scripts/06_export_onnx.py"),
    ("07", "scripts/07_inference.py"),
    ("07b", "scripts/07b_onnx_inference.py"),
    ("08", "scripts/08_benchmark.py"),
    ("09", "scripts/09_make_figures.py"),
    ("10", "scripts/10_make_report.py"),
    ("12", "scripts/12_package.py"),             # package first, so step 11 can also check the ZIP
    ("11", "scripts/11_verify_deliverables.py"),
]

# extra arguments for the two demo steps, so they do something useful without a user image
DEMO_ARGS = {
    "07": ["--model", "all", "--num-samples", "8"],
    "07b": ["--model", "all", "--num-samples", "200"],
}
# these steps have no --quick flag
NO_QUICK = {"07", "07b", "09", "10", "12"}


def run_step(step_id, script, quick):
    cmd = [sys.executable, "-u", str(ROOT / script)] + DEMO_ARGS.get(step_id, [])
    if quick and step_id not in NO_QUICK:
        cmd.append("--quick")
    log_path = LOGS / f"{Path(script).stem}.log"
    print("\n" + "=" * 70, flush=True)
    print(f"[{step_id}] python {script} {' '.join(cmd[3:])}", flush=True)
    print(f"    log: {log_path}", flush=True)
    print("=" * 70, flush=True)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ python {script} {' '.join(cmd[3:])}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    minutes = (time.time() - start) / 60
    print(f"[{step_id}] finished in {minutes:.1f} min with exit code {proc.returncode}", flush=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="smoke test with tiny runs")
    mode.add_argument("--full", action="store_true", help="the real run")
    parser.add_argument("--from", dest="start", default="01", help="step id to start from, e.g. 04")
    parser.add_argument("--to", dest="stop", default="11", help="last step id to run")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    ids = [s for s, _ in STEPS]
    first, last = ids.index(args.start), ids.index(args.stop)
    for step_id, script in STEPS[first:last + 1]:
        code = run_step(step_id, script, args.quick)
        if code != 0:
            print(f"\nStep {step_id} failed. Fix it and resume with: python run_all.py "
                  f"{'--quick' if args.quick else '--full'} --from {step_id}")
            sys.exit(code)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
