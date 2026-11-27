"""Ekstrak HOG dan histogram warna, lalu latih Random Forest."""

from __future__ import annotations

import argparse
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from src.eksperimen import informasi_data, sidik_data, simpan_hasil
from src.protokol import data


def hog(gambar):
    abu = np.asarray(gambar.convert("L"), dtype=np.float32) / 255.0
    dy, dx = np.gradient(abu)
    arah = np.mod(np.arctan2(dy, dx), np.pi) * (9 / np.pi)
    bawah = np.floor(arah).astype(np.int64)
    fraksi = arah - bawah
    magnitudo = np.hypot(dx, dy)
    y, x = np.indices(abu.shape)
    awal_sel = ((y // 8) * 8 + x // 8) * 9
    suara = np.bincount((awal_sel + bawah % 9).ravel(),
                        (magnitudo * (1 - fraksi)).ravel(), minlength=8 * 8 * 9)
    suara += np.bincount((awal_sel + (bawah + 1) % 9).ravel(),
                         (magnitudo * fraksi).ravel(), minlength=8 * 8 * 9)
    blok = sliding_window_view(suara.reshape(8, 8, 9), (2, 2), axis=(0, 1)).reshape(7, 7, 36)
    blok = blok / np.sqrt(np.sum(blok**2, axis=-1, keepdims=True) + 1e-6)
    blok = np.minimum(blok, 0.2)
    return (blok / np.sqrt(np.sum(blok**2, axis=-1, keepdims=True) + 1e-6)).ravel()


def fitur(sampel):
    vektor = []
    for x in sampel:
        gambar = Image.fromarray(x).resize((64, 64), Image.Resampling.BILINEAR)
        rgb = np.asarray(gambar)
        warna = np.concatenate([
            np.bincount((kanal // 16).ravel(), minlength=16) / kanal.size
            for kanal in rgb.transpose(2, 0, 1)
        ])
        vektor.append(np.concatenate((hog(gambar), warna)))
    return np.asarray(vektor, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pohon", type=int, default=200)
    parser.add_argument("--perangkat", default="cpu laptop")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.pohon < 1:
        parser.error("Seed di [0, 2³²) dan jumlah pohon minimal 1.")
    if args.final and informasi_data()["jenis"] != "primer":
        parser.error("Uji final memerlukan data primer.")
    t0 = time.perf_counter()
    data_hash = sidik_data()
    xtr, ytr, xva, yva, xte, yte = data.muat()
    latih = fitur(xtr)
    model = RandomForestClassifier(
        n_estimators=args.pohon, random_state=args.seed, class_weight="balanced", n_jobs=2
    )
    model.fit(latih, ytr)
    val = model.predict(fitur(xva)), yva
    uji = (model.predict(fitur(xte)), yte) if args.final else None
    simpan_hasil("baseline", "rf_hog_color", vars(args), data_hash,
                 time.perf_counter() - t0, val, uji, {"jumlah_fitur": latih.shape[1]})


if __name__ == "__main__":
    main()
