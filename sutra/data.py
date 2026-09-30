"""Mini-batching and dataset downloaders (MNIST, Tiny Shakespeare). Downloads are cached under ./data."""
from __future__ import annotations

import gzip
import os
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .tensor import Tensor, rng

DATA_DIR = Path(os.environ.get("SUTRA_DATA", Path(__file__).resolve().parent.parent / "data"))


class DataLoader:
    """Iterate over aligned arrays in (optionally shuffled) mini-batches, yielding Tensors."""

    def __init__(self, *arrays, batch_size: int = 32, shuffle: bool = False, drop_last: bool = False,
                 transform=None):
        n = len(arrays[0])
        assert all(len(a) == n for a in arrays), "arrays must share their first dimension"
        self.arrays, self.batch_size, self.shuffle, self.drop_last = arrays, batch_size, shuffle, drop_last
        self.transform = transform

    def __len__(self):
        n = len(self.arrays[0])
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.arrays[0])
        order = rng().permutation(n) if self.shuffle else np.arange(n)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            batch = [a[idx] for a in self.arrays]
            if self.transform is not None:
                batch = self.transform(*batch)
            yield tuple(Tensor(b) for b in batch)


def _fetch(url: str, tmp: Path) -> None:
    """urllib with certifi's CA bundle when present; fall back to curl (python.org macOS builds lack CA certs)."""
    try:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except (urllib.error.URLError, ssl.SSLError):
        subprocess.run(["curl", "-fsSL", "-o", str(tmp), url], check=True)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        _fetch(url, tmp)
        tmp.rename(dest)
    return dest


_MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def mnist():
    """Returns (x_train, y_train, x_test, y_test); images as uint8 (N, 28, 28), labels int64."""
    cache = DATA_DIR / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    out = {}
    for key, fname in _MNIST_FILES.items():
        dest = DATA_DIR / "mnist" / fname
        for mirror in _MNIST_MIRRORS:
            try:
                _download(mirror + fname, dest)
                break
            except Exception as e:  # try the next mirror
                print(f"  mirror failed: {e}")
        out[key] = _read_idx(dest)
    out["y_train"] = out["y_train"].astype(np.int64)
    out["y_test"] = out["y_test"].astype(np.int64)
    np.savez_compressed(cache, **out)
    return out["x_train"], out["y_train"], out["x_test"], out["y_test"]


def tiny_shakespeare() -> str:
    path = _download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                     DATA_DIR / "tinyshakespeare.txt")
    return path.read_text(encoding="utf-8")


def get_lm_batch(data: np.ndarray, batch_size: int, block_size: int):
    """Random contiguous windows from a token stream: inputs and next-token targets."""
    ix = rng().integers(0, len(data) - block_size - 1, size=batch_size)
    offs = ix[:, None] + np.arange(block_size)[None, :]
    return data[offs], data[offs + 1]
