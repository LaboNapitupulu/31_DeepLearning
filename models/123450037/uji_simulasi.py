"""Jalankan tugas #13 di salinan kode dengan data dan artefak terpisah."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

AKAR = Path(__file__).resolve().parents[2]


def sidik(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jalankan_simulasi(args):
    # Evaluator mencetak Unicode; beberapa terminal Windows masih memakai cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if not 0 <= args.seed < 2**32 or min(args.epochs, args.batch_size, args.threads) < 1:
        raise SystemExit("Seed, epoch, batch, atau thread tidak sah.")
    if (not np.isfinite([args.lr, args.weight_decay, args.dropout]).all()
            or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        raise SystemExit("LR, weight decay, atau dropout tidak sah.")
    bobot = str(Path(args.bobot).resolve()) if args.bobot else None
    if bobot and not Path(bobot).is_file():
        raise SystemExit(f"Berkas bobot tidak ditemukan: {bobot}")

    # Sidik input primer dan seluruh artefak anggota yang sudah ada.
    dilindungi = [AKAR / "data/MANIFEST.json"]
    dilindungi.extend(p for p in (AKAR / "data/okupansi").glob("*") if p.is_file())
    dilindungi.extend(p for p in (AKAR / "results/123450037").rglob("*") if p.is_file())
    sebelum = {str(p.relative_to(AKAR)): sidik(p) for p in dilindungi}

    hasil = AKAR / "results/simulasi/123450037"
    hasil.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"seed_{args.seed}-", dir=hasil))
    shutil.copytree(AKAR / "src", run / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(AKAR / "models/123450037", run / "models/123450037",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (run / "data").mkdir()
    shutil.copyfile(AKAR / "data/MANIFEST.json", run / "data/MANIFEST.json")
    (run / ".simulasi-sprint").write_text("Simulasi tugas #13\n", encoding="utf-8")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_HOME"] = str(AKAR / ".venv/torch-cache")
    perintah = [sys.executable, "-B", "models/123450037/run.py"]
    for name in ("seed", "epochs", "batch_size", "lr", "weight_decay", "dropout", "threads", "perangkat"):
        perintah.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if bobot:
        perintah.extend(["--bobot", bobot])
    if args.bobot_acak_simulasi:
        perintah.append("--bobot-acak-simulasi")
    if args.fine_tune:
        perintah.append("--fine-tune")

    print(f"Folder simulasi terpisah: {run}", flush=True)
    with (run / "run.log").open("w", encoding="utf-8") as log:
        for cmd in ([sys.executable, "-B", "-m", "src.protokol.simulasi"], perintah):
            line = "$ " + subprocess.list2cmdline(cmd)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            with subprocess.Popen(cmd, cwd=run, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8") as proses:
                for line in proses.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if proses.wait():
                    raise SystemExit(f"Simulasi gagal; periksa {run / 'run.log'}")

    artefak = run / "results/123450037"
    metrics = json.loads((artefak / "metrics.json").read_text("utf-8"))
    detail = json.loads((artefak / "detail.json").read_text("utf-8"))
    provenance = json.loads((run / "data/okupansi/provenance.json").read_text("utf-8"))
    pred = np.load(artefak / "pred_val.npy", allow_pickle=False)
    with np.load(run / "data/okupansi/dataset.npz", allow_pickle=False) as dataset:
        assert dataset["X"].shape[1:] == (224, 224, 3)
        assert dataset["X"].dtype == np.uint8
        y_val = dataset["y"][dataset["val"]]
        assert pred.shape == y_val.shape == (80,)
        assert np.issubdtype(pred.dtype, np.integer) and np.isin(pred, range(4)).all()
    from src.common import evaluate

    hitung = evaluate.metrik_klasifikasi(y_val, pred)
    assert round(hitung["acc"], 6) == metrics["metrik"]["val_acc"]
    assert round(hitung["macro_f1"], 6) == metrics["metrik"]["val_macro_f1"]
    assert np.isclose(np.abs(pred - y_val).mean(), detail["metrik_tambahan"]["val_mae"])
    assert metrics["seed"] == args.seed and metrics["model"] == "tl"
    assert provenance["jenis"] == detail["jenis_data"] == "simulasi"
    assert provenance["ukuran_split"] == {"train": 160, "val": 80, "test": 80}
    assert provenance["sesi"][0]["jumlah_kondisi"] == 160
    kelompok = [set(provenance["kelompok_split"][s]) for s in ("train", "val", "test")]
    assert all(not kelompok[i] & kelompok[j] for i in range(3) for j in range(i + 1, 3))
    assert metrics["n_test"] == 0 and not (artefak / "pred_test.npy").exists()
    assert metrics["data_hash"] == detail["data_hash"]
    assert (artefak / f"seed_{args.seed}/model.pt").is_file()
    assert all((AKAR / p).is_file() and sidik(AKAR / p) == h for p, h in sebelum.items())

    ringkasan = {
        "tugas": 13, "status": "berhasil", "jenis_data": "simulasi",
        "data_hash": metrics["data_hash"], "seed": args.seed,
        "bobot_awal": "acak" if args.bobot_acak_simulasi else "ImageNet",
        "sha256_bobot": sidik(Path(bobot)) if bobot else None,
        "lingkungan": {"python": sys.version.split()[0],
                       **{p: version(p) for p in ("torch", "torchvision", "numpy", "Pillow")}},
        "ukuran_split": provenance["ukuran_split"], "kelompok_split": provenance["kelompok_split"],
        "metrik_validasi": {**metrics["metrik"], **detail["metrik_tambahan"]},
        "durasi_detik": metrics["durasi_detik"],
        "data_dan_artefak_primer_tetap_sama": True, "data_uji_dievaluasi": False,
        "folder_hasil": str(run.relative_to(AKAR)),
        "perintah_model": perintah[2:],
    }
    (run / "ringkasan.json").write_text(json.dumps(ringkasan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ringkasan, indent=2), flush=True)
