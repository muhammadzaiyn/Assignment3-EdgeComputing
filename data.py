"""CIFAR-10 download, train/val/test splits, data loaders and the calibration subset.

The data is always downloaded from code. We first try the official torchvision source
(University of Toronto). That server is sometimes very slow, so if the download speed is
too low after about 3 minutes we cancel it and use the Hugging Face copy instead.
Both paths end in the same numpy arrays, so the rest of the project does not care.
"""
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

SPLIT_SEED = 42
NUM_TRAIN = 45_000
NUM_VAL = 5_000
CALIB_SEED = 42


# ---------------------------------------------------------------- download

def _download_with_speed_check(url, dest, check_after_s=180, min_mb_per_min=3.0):
    """Download url to dest. Return False if the server is too slow (so we can fall back)."""
    tmp = dest.with_suffix(".part")
    start = time.time()
    already = tmp.stat().st_size if tmp.exists() else 0
    request = urllib.request.Request(url)
    if already:
        # resume an interrupted download instead of starting again
        request.add_header("Range", f"bytes={already}-")
    with urllib.request.urlopen(request, timeout=60) as resp:
        resumed = resp.status == 206
        mode = "ab" if resumed else "wb"
        if resumed:
            print(f"  resuming from {already / 1e6:.1f} MB")
        else:
            already = 0
        total = int(resp.headers.get("Content-Length", 0)) + already
        done = 0  # bytes downloaded in this session (used for the speed check)
        last_print = 0
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                elapsed = time.time() - start
                if elapsed - last_print > 10:
                    last_print = elapsed
                    print(f"  downloaded {(already + done) / 1e6:.1f} / {total / 1e6:.1f} MB in {elapsed:.0f} s")
                if elapsed > check_after_s:
                    mb_per_min = done / 1e6 / (elapsed / 60)
                    remaining_min = (total - already - done) / 1e6 / max(mb_per_min, 1e-6)
                    if mb_per_min < min_mb_per_min or remaining_min > 30:
                        print(f"  too slow ({mb_per_min:.2f} MB/min), giving up on this server")
                        return False
    tmp.replace(dest)
    return True


def _load_torchvision_batches(folder):
    """Read the extracted python pickle batches into uint8 arrays (N, 32, 32, 3)."""
    def read(name):
        with open(folder / name, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.asarray(d["labels"], dtype=np.int64)

    parts = [read(f"data_batch_{i}") for i in range(1, 6)]
    train_x = np.concatenate([p[0] for p in parts])
    train_y = np.concatenate([p[1] for p in parts])
    test_x, test_y = read("test_batch")
    return train_x, train_y, test_x, test_y


def _try_torchvision_source():
    """Official source. Uses torchvision for the md5 check and extraction."""
    from torchvision.datasets import CIFAR10

    folder = DATA_DIR / "cifar-10-batches-py"
    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    if not folder.exists():
        if not tar_path.exists():
            print(f"Downloading CIFAR-10 from {CIFAR_URL}")
            if not _download_with_speed_check(CIFAR_URL, tar_path):
                return None
        # download=True finds the tarball, checks the md5 and extracts it
        CIFAR10(root=str(DATA_DIR), train=True, download=True)
    return _load_torchvision_batches(folder)


def _try_huggingface_source():
    """Fallback: Hugging Face copy of CIFAR-10 (columns 'img' and 'label')."""
    from datasets import load_dataset

    print("Downloading CIFAR-10 from Hugging Face (uoft-cs/cifar10)")
    ds = load_dataset("uoft-cs/cifar10", cache_dir=str(DATA_DIR / "hf_cache"))
    assert set(ds["train"].column_names) >= {"img", "label"}, ds["train"].column_names

    def to_arrays(split):
        x = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in split["img"]])
        return x, np.asarray(split["label"], dtype=np.int64)

    train_x, train_y = to_arrays(ds["train"])
    test_x, test_y = to_arrays(ds["test"])
    return train_x, train_y, test_x, test_y


def load_cifar10_arrays():
    """Return (train_x, train_y, test_x, test_y). Cached in data/cifar10.npz after the first call."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / "cifar10.npz"
    if cache.exists():
        d = np.load(cache)
        return d["train_x"], d["train_y"], d["test_x"], d["test_y"]

    arrays, source = None, "torchvision (cs.toronto.edu)"
    try:
        arrays = _try_torchvision_source()
    except Exception as e:  # network errors, bad md5, ...
        print(f"  torchvision source failed: {e}")
    if arrays is None:
        arrays, source = _try_huggingface_source(), "Hugging Face (uoft-cs/cifar10)"

    train_x, train_y, test_x, test_y = arrays
    assert train_x.shape == (50_000, 32, 32, 3) and test_x.shape == (10_000, 32, 32, 3)
    np.savez(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, source=source)
    print(f"CIFAR-10 cached in {cache} (source: {source})")
    return train_x, train_y, test_x, test_y


def dataset_source():
    """Where the cached copy of CIFAR-10 was downloaded from."""
    d = np.load(DATA_DIR / "cifar10.npz")
    return str(d["source"]) if "source" in d.files else "unknown"


# ---------------------------------------------------------------- datasets

def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


class ArrayDataset(Dataset):
    """Images stored as a uint8 array (N, 32, 32, 3) plus integer labels."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return self.transform(img), int(self.labels[i])


def split_indices():
    """Fixed 45,000 / 5,000 split of the official training set."""
    perm = np.random.RandomState(SPLIT_SEED).permutation(NUM_TRAIN + NUM_VAL)
    return perm[:NUM_TRAIN], perm[NUM_TRAIN:]


def get_datasets(quick=False):
    """Return train, val and test datasets. Quick mode uses small subsets."""
    train_x, train_y, test_x, test_y = load_cifar10_arrays()
    train_idx, val_idx = split_indices()
    if quick:
        train_idx, val_idx = train_idx[:2000], val_idx[:500]
        test_x, test_y = test_x[:1000], test_y[:1000]
    train_ds = ArrayDataset(train_x[train_idx], train_y[train_idx], train_transform())
    val_ds = ArrayDataset(train_x[val_idx], train_y[val_idx], eval_transform())
    test_ds = ArrayDataset(test_x, test_y, eval_transform())
    return train_ds, val_ds, test_ds


def get_test_dataset(limit=None):
    _, _, test_x, test_y = load_cifar10_arrays()
    if limit:
        test_x, test_y = test_x[:limit], test_y[:limit]
    return ArrayDataset(test_x, test_y, eval_transform())


def get_calibration_dataset(num_images=1000, seed=CALIB_SEED):
    """A fixed random subset of the 45k training images, with the eval transform.

    Calibration uses training images without augmentation, so the activation ranges the
    observers see look like the ranges at test time.
    """
    train_x, train_y, _, _ = load_cifar10_arrays()
    train_idx, _ = split_indices()
    chosen = np.random.RandomState(seed).choice(train_idx, size=num_images, replace=False)
    return ArrayDataset(train_x[chosen], train_y[chosen], eval_transform())


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
