"""Build the submission ZIP (<ROLL_NO>_<SECTION>.zip) in the project root.

Includes: src/, scripts/, run_all.py, README.md, requirements.txt, the final checkpoints,
onnx_models/, results/, report/ and docs/VIVA_NOTES.md.
Excludes: data/, .venv/, __pycache__/, checkpoints/intermediate/, CLAUDE*.md.

Run:  python scripts/12_package.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLL_NO = "23I-0051"
SECTION = "AI-A"
ZIP_PATH = ROOT / f"{ROLL_NO}_{SECTION}.zip"
TOP = "edge_assignment"  # folder name inside the ZIP

FINAL_CHECKPOINTS = [
    "resnet18_fp32.pth",
    "resnet18_ptq_int8.pt",
    "resnet18_ptq_int8_state.pth",   # same PTQ model as state_dict + config
    "resnet18_qat_int8.pt",
    "resnet18_qat_int8_state.pth",   # same QAT model as state_dict + config
    "resnet18_unstructured.pth",
    "resnet18_unstructured.pth.gz",  # the same 90 %-sparse model, gzip-compressed for storage
    "resnet18_structured.pth",
]
FOLDERS = ["src", "scripts", "onnx_models", "results", "report", "docs"]
FILES = ["run_all.py", "README.md", "requirements.txt"]
SKIP_PARTS = {"__pycache__", ".venv", "data", "intermediate", ".ipynb_checkpoints"}


def files_to_pack():
    files = [ROOT / f for f in FILES]
    files += [ROOT / "checkpoints" / f for f in FINAL_CHECKPOINTS]
    for folder in FOLDERS:
        for f in sorted((ROOT / folder).rglob("*")):
            rel = f.relative_to(ROOT)
            if f.is_file() and not SKIP_PARTS.intersection(rel.parts) and f.suffix != ".pyc" \
                    and not f.name.startswith("~$"):
                files.append(f)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"Missing files, not packaging: {missing}")
    return files


def main():
    files = files_to_pack()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"{TOP}/{f.relative_to(ROOT).as_posix()}")
    size = ZIP_PATH.stat().st_size / 1e6
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
    for n in names:
        print(n)
    print(f"\n{ZIP_PATH.name}: {len(names)} files, {size:.1f} MB")
    if size > 300:
        print("The ZIP is over 300 MB. The ONNX copies of the float models are the easiest thing to trim.")


if __name__ == "__main__":
    main()
