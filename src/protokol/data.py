"""Muat citra RGB uint8 dan label okupansi 0–3 dengan split beku per kelompok."""

import numpy as np

from src.common.manifest import AKAR

TUGAS = "klasifikasi"


def muat():
    path = AKAR / "data" / "okupansi" / "dataset.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: siapkan arsip melalui python -m src.protokol.siapkan."
        )
    with np.load(path, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        return tuple(
            array[data[bagian]]
            for bagian in ("train", "val", "test")
            for array in (X, y)
        )
