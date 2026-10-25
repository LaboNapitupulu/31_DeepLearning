"""Pemuat Fashion-MNIST dari berkas IDX resmi (milik dosen, jangan diubah).

Data dicari di beberapa lokasi, urut: variabel lingkungan ``DSW_DATA_RAW``, ``../../data/raw``,
``../data/raw``, ``data/raw`` (relatif terhadap akar repo), lalu workspace Workbench
``~/workbench-workspace/deep-learning/data/raw``. Struktur mengikuti torchvision:
``FashionMNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte``.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
KELAS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
_BERKAS = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}


def folder_data() -> Path:
    kandidat = []
    if os.environ.get("DSW_DATA_RAW"):
        kandidat.append(Path(os.environ["DSW_DATA_RAW"]))
    kandidat += [AKAR.parent.parent / "data" / "raw", AKAR.parent / "data" / "raw", AKAR / "data" / "raw",
                 Path.home() / "workbench-workspace" / "deep-learning" / "data" / "raw"]
    for k in kandidat:
        if (k / "FashionMNIST" / "raw" / _BERKAS["train"][1]).exists() or \
           (k / "FashionMNIST" / "raw" / (_BERKAS["train"][1] + ".gz")).exists():
            return k
    raise FileNotFoundError("Fashion-MNIST tidak ditemukan. Siapkan dataset Modul 05 dari Workbench "
                            "atau setel DSW_DATA_RAW ke folder data/raw.")


def _baca(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gzip.decompress(gz.read_bytes())
    raise FileNotFoundError(path)


def jalur_label(bagian: str) -> Path:
    return folder_data() / "FashionMNIST" / "raw" / _BERKAS[bagian][1]


def muat(bagian: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """``(X uint8 [N,28,28], y int64 [N])`` untuk ``train`` (60.000) atau ``test`` (10.000)."""
    d = folder_data() / "FashionMNIST" / "raw"
    img_b = _baca(d / _BERKAS[bagian][0])
    lbl_b = _baca(d / _BERKAS[bagian][1])
    n = int.from_bytes(img_b[4:8], "big")
    X = np.frombuffer(img_b, dtype=np.uint8, offset=16).reshape(n, 28, 28)
    y = np.frombuffer(lbl_b, dtype=np.uint8, offset=8).astype(np.int64)
    return X, y
