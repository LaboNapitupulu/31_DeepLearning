"""Kerangka run.py anggota. Salin folder ini menjadi models/<nim>/ lalu isi bagian MODEL.

    python models/<nim>/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common import data, split  # noqa: E402

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None). Ganti isi fungsi ini dengan model Anda."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    d = split.muat_split()
    Xtr, ytr = d["train"]
    Xva, _ = d["val"]
    # ---- MODEL (contoh: regresi logistik; ganti dengan arsitektur jalur Anda) ----
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)          # tulis alasan pilihan optimizer di catatan.md
    X = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(ytr)
    for epoch in range(2):
        for i in range(0, len(X), 128):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(X[i:i + 128]), y[i:i + 128])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.tensor(Xva, dtype=torch.float32).unsqueeze(1) / 255.0).argmax(1).numpy()
        pred_test = None
        if final:
            Xte, _ = data.muat("test")
            pred_test = model(torch.tensor(Xte, dtype=torch.float32).unsqueeze(1) / 255.0).argmax(1).numpy()
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val, dtype=np.int64))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test, dtype=np.int64))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
