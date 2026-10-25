"""Baseline dosen: MLP 784→128→10 dengan NumPy murni (milik dosen, jangan diubah).

    python -m src.common.baseline_mlp --seed 2026

Batas bawah yang harus dilampaui setiap model. Dilatih 3 epoch SGD mini-batch pada split
latih beku, dievaluasi pada split validasi, lalu ditulis ke ``results/baseline/metrics.json``
lewat ``evaluate`` (jalur yang sama dengan model mahasiswa).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import data, split


def latih(seed: int, epoch: int = 3, lr: float = 0.1, batch: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = split.muat_split()
    Xtr = d["train"][0].reshape(-1, 784).astype(np.float32) / 255.0
    ytr = d["train"][1]
    Xva = d["val"][0].reshape(-1, 784).astype(np.float32) / 255.0
    W1 = rng.normal(0, np.sqrt(2 / 784), (784, 128)).astype(np.float32); b1 = np.zeros(128, np.float32)
    W2 = rng.normal(0, np.sqrt(2 / 128), (128, 10)).astype(np.float32); b2 = np.zeros(10, np.float32)
    n = len(Xtr)
    for ep in range(epoch):
        urut = rng.permutation(n)
        for i in range(0, n, batch):
            idx = urut[i:i + batch]
            x, y = Xtr[idx], ytr[idx]
            h = np.maximum(x @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(y)), y] -= 1
            p /= len(y)
            gW2 = h.T @ p; gb2 = p.sum(0)
            gh = (p @ W2.T) * (h > 0)
            gW1 = x.T @ gh; gb1 = gh.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
        acc = ((np.maximum(Xva @ W1 + b1, 0) @ W2 + b2).argmax(1) == d["val"][1]).mean()
        print(f"epoch {ep + 1}: val_acc {acc:.4f}")
    return (np.maximum(Xva @ W1 + b1, 0) @ W2 + b2).argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    t0 = time.time()
    pred = latih(a.seed)
    durasi = time.time() - t0
    out = data.AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred.astype(np.int64))
    subprocess.run([sys.executable, "-m", "src.common.evaluate", "--nim", "baseline", "--model", "baseline",
                    "--seed", str(a.seed), "--pred-val", str(out / "pred_val.npy"),
                    "--durasi", f"{durasi:.1f}", "--perangkat", "cpu laptop"], check=True, cwd=data.AKAR)


if __name__ == "__main__":
    main()
