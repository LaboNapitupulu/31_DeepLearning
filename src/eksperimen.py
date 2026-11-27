"""Simpan hasil model tim lewat evaluator starter dan catat MAE okupansi."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np

from src.common import manifest, split


def sidik_data():
    return manifest.data_hash(f"split:{split.SEED_SPLIT}".encode())


def informasi_data():
    return json.loads((manifest.AKAR / "data/okupansi/provenance.json").read_text("utf-8"))


def simpan_hasil(nim, jalur, konfigurasi, data_hash, durasi, validasi, uji=None, detail=None):
    seed = konfigurasi["seed"]
    akar = manifest.AKAR / "results" / nim
    run = akar / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    pred_val, y_val = validasi
    np.save(run / "pred_val.npy", pred_val)
    perintah = [
        sys.executable, "-m", "src.common.evaluate", "--nim", nim, "--model", jalur,
        "--seed", str(seed), "--pred-val", str(run / "pred_val.npy"),
        "--durasi", f"{durasi:.3f}", "--perangkat", konfigurasi["perangkat"],
        "--out", str(run / "metrics.json"),
    ]
    metrik = {"val_mae": float(np.abs(pred_val - y_val).mean())}
    if uji is not None:
        pred_test, y_test = uji
        np.save(run / "pred_test.npy", pred_test)
        perintah.extend(["--pred-test", str(run / "pred_test.npy")])
        metrik["test_mae"] = float(np.abs(pred_test - y_test).mean())
    else:
        (run / "pred_test.npy").unlink(missing_ok=True)
    subprocess.run(perintah, check=True, cwd=manifest.AKAR)
    hasil = json.loads((run / "metrics.json").read_text("utf-8"))
    if hasil["data_hash"] != data_hash:
        raise SystemExit("Data berubah selama pelatihan. Jalankan lagi pada data yang sama.")
    catatan = {
        "seed": seed, "data_hash": data_hash, "jenis_data": informasi_data()["jenis"],
        "metrik_tambahan": metrik, "konfigurasi": konfigurasi, **(detail or {}),
    }
    (run / "detail.json").write_text(json.dumps(catatan, indent=2) + "\n", encoding="utf-8")
    for nama in ("pred_val.npy", "detail.json", "metrics.json"):
        shutil.copyfile(run / nama, akar / nama)
    if uji is not None:
        shutil.copyfile(run / "pred_test.npy", akar / "pred_test.npy")
    else:
        (akar / "pred_test.npy").unlink(missing_ok=True)
    print(f"{nim}, seed {seed}: MAE {metrik}", flush=True)
    return run
