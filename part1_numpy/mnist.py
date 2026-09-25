"""Download and load MNIST as NumPy arrays."""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
IMAGES_MAGIC = 2051
LABELS_MAGIC = 2049


def download_mnist(data_dir: Path = DATA_DIR) -> None:
    """Download the four MNIST files into `data_dir`, skipping any already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = data_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(BASE_URL + filename, path)


def _read_idx(path: Path, magic: int, header_ints: int) -> np.ndarray:
    """Read a gzipped IDX file: a big-endian int32 header followed by uint8 data."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    header = np.frombuffer(raw, dtype=">i4", count=header_ints)
    if header[0] != magic:
        raise ValueError(f"{path.name}: expected magic number {magic}, got {header[0]}")
    return np.frombuffer(raw, dtype=np.uint8, offset=4 * header_ints)


def load_mnist(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test).

    X arrays are float32 of shape (N, 784) with pixel values scaled to [0, 1].
    y arrays are int64 of shape (N,) with digit labels 0-9.
    """
    download_mnist(data_dir)

    def images(key: str) -> np.ndarray:
        data = _read_idx(data_dir / FILES[key], IMAGES_MAGIC, header_ints=4)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    def labels(key: str) -> np.ndarray:
        return _read_idx(data_dir / FILES[key], LABELS_MAGIC, header_ints=2).astype(np.int64)

    return images("train_images"), labels("train_labels"), images("test_images"), labels("test_labels")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train: {X_train.shape} {y_train.shape}  test: {X_test.shape} {y_test.shape}")
