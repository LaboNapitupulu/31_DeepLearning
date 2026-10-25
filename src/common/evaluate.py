"""Penilai hasil model (milik dosen, jangan diubah).

    python -m src.common.evaluate --nim 12345678 --model cnn --seed 1 \
        --pred-val results/12345678/pred_val.npy --durasi 540 --perangkat "cpu laptop"
    # sprint terakhir, SEKALI, oleh Penjaga Data:
    python -m src.common.evaluate --nim 12345678 --model cnn --seed 1 \
        --pred-val ... --pred-test results/12345678/pred_test.npy ...

Menulis ``results/<nim>/metrics.json`` dengan skema baku Workbench:
``metrik`` (angka), ``seed`` (bulat), ``data_hash`` (heksadesimal), ``durasi_detik``,
``perangkat``; ditambah ``model`` dan ``n_val``/``n_test`` sebagai keterangan.
Prediksi berupa array ``int`` label kelas 0–9 sepanjang data validasi (6.000) atau uji (10.000).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data, split


def akurasi(y: np.ndarray, p: np.ndarray) -> float:
    return float((y == p).mean())


def macro_f1(y: np.ndarray, p: np.ndarray, k: int = 10) -> float:
    f1 = []
    for c in range(k):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def _muat_pred(path: str, n: int) -> np.ndarray:
    p = np.load(path)
    if p.ndim == 2:          # logits/probabilitas → argmax
        p = p.argmax(axis=1)
    p = p.astype(np.int64).ravel()
    if p.shape[0] != n:
        raise SystemExit(f"{path}: panjang {p.shape[0]}, seharusnya {n}")
    if p.min() < 0 or p.max() > 9:
        raise SystemExit(f"{path}: label harus 0–9")
    return p


def nilai(pred_val: str, pred_test: str | None = None) -> dict[str, float]:
    _, y_tr = data.muat("train")
    _, idx_val = split.indeks()
    y_val = y_tr[idx_val]
    p_val = _muat_pred(pred_val, len(y_val))
    metrik = {"val_acc": akurasi(y_val, p_val), "val_macro_f1": macro_f1(y_val, p_val)}
    if pred_test:
        _, y_te = data.muat("test")
        p_te = _muat_pred(pred_test, len(y_te))
        metrik.update(test_acc=akurasi(y_te, p_te), test_macro_f1=macro_f1(y_te, p_te))
    return metrik


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nim", required=True)
    ap.add_argument("--model", required=True, help="jalur: cnn | tl | rnn | trf | ae | baseline")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pred-val", required=True)
    ap.add_argument("--pred-test", default=None, help="hanya pada sprint terakhir, sekali")
    ap.add_argument("--durasi", type=float, default=None, help="detik pelatihan")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--out", default=None, help="bawaan results/<nim>/metrics.json")
    a = ap.parse_args()

    metrik = nilai(a.pred_val, a.pred_test)
    hasil = {"metrik": {k: round(v, 6) for k, v in metrik.items()}, "seed": int(a.seed),
             "data_hash": split.data_hash(), "perangkat": a.perangkat, "model": a.model,
             "n_val": split.N_VAL, "n_test": 10000 if a.pred_test else 0}
    if a.durasi is not None:
        hasil["durasi_detik"] = float(a.durasi)
    out = Path(a.out) if a.out else data.AKAR / "results" / a.nim / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(hasil["metrik"]), "→", out.relative_to(data.AKAR) if out.is_relative_to(data.AKAR) else out)


if __name__ == "__main__":
    main()
