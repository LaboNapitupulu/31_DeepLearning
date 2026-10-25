"""Split latih/validasi yang dibekukan (milik dosen, jangan diubah).

Permutasi 60.000 indeks data latih resmi dengan ``numpy.random.default_rng(2026)``;
6.000 pertama = validasi, sisanya latih. Sama di semua laptop dan semua tim.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import data

SEED_SPLIT = 2026
N_VAL = 6000


def indeks() -> tuple[np.ndarray, np.ndarray]:
    """``(idx_latih, idx_val)`` atas data latih resmi."""
    urut = np.random.default_rng(SEED_SPLIT).permutation(60000)
    return urut[N_VAL:], urut[:N_VAL]


def muat_split() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{"train": (X, y), "val": (X, y)}``; data uji resmi tidak ikut (lihat evaluate)."""
    X, y = data.muat("train")
    tr, va = indeks()
    return {"train": (X[tr], y[tr]), "val": (X[va], y[va])}


def data_hash() -> str:
    """Sidik data yang dipakai: berkas label latih + uji + indeks validasi (hex 32)."""
    h = hashlib.sha256()
    for bagian in ("train", "test"):
        h.update(data._baca(data.jalur_label(bagian)))
    h.update(indeks()[1].astype(np.int64).tobytes())
    return h.hexdigest()[:32]
