"""Minimal QMNIST dataset loader (no download).

Reads directly from the pre-cloned repository at C:/Users/w105336/source/qmnist.
Returns (flat float32 tensor of shape [784], int class label) per sample,
ready for use with torch.utils.data.DataLoader.
"""

import codecs
import gzip
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_QMNIST_SOURCE = r"../qmnist"

_FILES = {
    "train": (
        "qmnist-train-images-idx3-ubyte.gz",
        "qmnist-train-labels-idx2-int.gz",
    ),
    "test": (
        "qmnist-test-images-idx3-ubyte.gz",
        "qmnist-test-labels-idx2-int.gz",
    ),
}


def _get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def _read_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert _get_int(data[:4]) == 8 * 256 + 3, "Not an idx3-ubyte file"
    length = _get_int(data[4:8])
    rows = _get_int(data[8:12])
    cols = _get_int(data[12:16])
    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
    return torch.from_numpy(parsed.copy()).view(length, rows, cols)


def _read_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    assert _get_int(data[:4]) == 12 * 256 + 2, "Not an idx2-int file"
    length = _get_int(data[4:8])
    width = _get_int(data[8:12])
    parsed = np.frombuffer(data, dtype=np.dtype(">i4"), offset=12)
    return torch.from_numpy(parsed.copy().astype("i4")).view(length, width).long()


class QMNIST(Dataset):
    """QMNIST dataset loaded from a local clone of the repository.

    Args:
        what (str): One of ``'train'``, ``'test'``, or ``'test10k'``.
            ``'test10k'`` returns the first 10 000 test examples (mirrors the
            classic MNIST test set).
        train (bool): Convenience alias — when *what* is not given, ``True``
            selects ``'train'`` and ``False`` selects ``'test'``.
        source_dir (str): Path to the local QMNIST clone.  Defaults to the
            hard-coded path set in this module.
    """

    classes = [
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine",
    ]

    def __init__(
        self,
        what: str | None = None,
        train: bool = True,
        source_dir: str = _QMNIST_SOURCE,
    ) -> None:
        if what is None:
            what = "train" if train else "test"
        if what not in ("train", "test", "test10k"):
            raise ValueError(f"'what' must be 'train', 'test', or 'test10k'; got {what!r}")

        subset = "test" if what in ("test", "test10k") else "train"
        img_file, lbl_file = _FILES[subset]

        img_path = os.path.join(source_dir, img_file)
        lbl_path = os.path.join(source_dir, lbl_file)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.isfile(lbl_path):
            raise FileNotFoundError(f"Label file not found: {lbl_path}")

        images = _read_images(img_path)   # uint8, shape [N, 28, 28]
        labels = _read_labels(lbl_path)   # long, shape [N, 8]

        if what == "test10k":
            images = images[:10_000]
            labels = labels[:10_000]

        # Flatten to [N, 784] and normalise to [0, 1]
        self.data: torch.Tensor = images.view(len(images), -1).float() / 255.0
        # Column 0 is the digit class
        self.targets: torch.Tensor = labels[:, 0]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index].item()
