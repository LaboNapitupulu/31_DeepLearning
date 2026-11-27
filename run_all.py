"""Latih semua jalur okupansi dan buat tabel komparasi Sprint 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from src.eksperimen import informasi_data, sidik_data

AKAR = Path(__file__).resolve().parent
ANGGOTA = {
    "123450037": "tl",
    "123450040": "cnn",
    "123450078": "trf",
    "123450091": "rnn",
}


def jalankan(opsi, cwd=AKAR):
    perintah = [sys.executable, *opsi]
    print("$", " ".join(perintah), flush=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "4"}
    subprocess.run(perintah, check=True, cwd=cwd, env=env)


def baca_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def baca_hasil(nim, jalur, seed, hash_data):
    folder = AKAR / "results" / nim
    hasil = baca_json(folder / "metrics.json")
    detail = baca_json(folder / "detail.json")
    if (hasil["seed"] != seed or hasil["model"] != jalur or hasil["data_hash"] != hash_data
            or detail["data_hash"] != hash_data or detail["seed"] != seed):
        raise RuntimeError(f"Artefak {nim} tidak cocok dengan model, seed, atau data run saat ini.")
    hasil["metrik"].update(detail["metrik_tambahan"])
    return hasil


def baseline_starter(seed, hash_data):
    with tempfile.TemporaryDirectory(prefix="okupansi-starter-") as temp:
        akar = Path(temp)
        shutil.copytree(AKAR / "src", akar / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (akar / "data").symlink_to(AKAR / "data", target_is_directory=True)
        jalankan(["-m", "src.common.baseline", "--seed", str(seed)], cwd=akar)
        hasil = baca_json(akar / "results/baseline/metrics.json")
        if hasil["data_hash"] != hash_data:
            raise RuntimeError("Data baseline starter berbeda dengan data model tim.")
        pred = np.load(akar / "results/baseline/pred_val.npy", allow_pickle=False)
        with np.load(akar / "data/okupansi/dataset.npz", allow_pickle=False) as arsip:
            hasil["metrik"]["val_mae"] = float(np.abs(pred - arsip["y"][arsip["val"]]).mean())
        shutil.copytree(akar / "results/baseline", AKAR / "results/baseline_starter", dirs_exist_ok=True)
    return hasil


def tabel(baris, data, hash_data):
    sumber = data["jenis"]
    isi = [
        f"# Komparasi okupansi · data {sumber}",
        "",
        f"Sidik data: `{hash_data}`.",
        f"Jumlah citra latih/validasi/uji: {' / '.join(map(str, data['ukuran_split'].values()))}.",
        "",
        "| Komponen | Jalur | Seed | Val akurasi (%) | Val macro-F1 | Val MAE | Test akurasi (%) | Test MAE | Detik |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nim, h in baris:
        m = h["metrik"]
        val_mae = f"{m['val_mae']:.4f}" if "val_mae" in m else "—"
        test_acc = f"{100 * m['test_acc']:.2f}" if "test_acc" in m else "—"
        test_mae = f"{m['test_mae']:.4f}" if "test_mae" in m else "—"
        isi.append(f"| {nim} | {h['model']} | {h['seed']} | {100 * m['val_acc']:.2f} | "
                   f"{m['val_macro_f1']:.4f} | {val_mae} | {test_acc} | {test_mae} | "
                   f"{h.get('durasi_detik', 0):.2f} |")
    if sumber == "simulasi":
        isi.extend(["", "Hasil dari citra sintetis untuk memeriksa alur eksekusi."])
    path = AKAR / "reports" / "hasil_komparasi.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(isi) + "\n", encoding="utf-8")
    print(f"Tabel komparasi: {path}", flush=True)


def reproduksi_simulasi(args):
    opsi = ["run_all.py", "--_simulasi-terisolasi", "--seeds", *map(str, args.seeds)]
    if args.hanya:
        opsi.extend(["--hanya", *args.hanya])
    if args.tanpa_baseline:
        opsi.append("--tanpa-baseline")
    if args.bobot_tl:
        opsi.extend(["--bobot-tl", str(args.bobot_tl.resolve())])
    with tempfile.TemporaryDirectory(prefix="okupansi-simulasi-") as folder:
        akar = Path(folder)
        for name in ("src", "models"):
            shutil.copytree(AKAR / name, akar / name, ignore=shutil.ignore_patterns("__pycache__"))
        (akar / "data").mkdir()
        (akar / ".simulasi-sprint").touch()
        shutil.copyfile(AKAR / "data/MANIFEST.json", akar / "data/MANIFEST.json")
        shutil.copyfile(AKAR / "run_all.py", akar / "run_all.py")
        jalankan(["-m", "src.protokol.simulasi"], cwd=akar)
        jalankan(opsi, cwd=akar)
        tujuan = AKAR / "results/simulasi"
        if args.hanya or args.tanpa_baseline:
            bagian = "-".join(args.hanya or ("semua-jalur",))
            tujuan /= f"uji-{bagian}"
        shutil.copytree(akar / "results", tujuan, dirs_exist_ok=True)
        shutil.copyfile(akar / "reports/hasil_komparasi.md", tujuan / "hasil_komparasi.md")
        print(f"Artefak simulasi: {tujuan}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--hanya", nargs="+", choices=tuple(ANGGOTA))
    parser.add_argument("--tanpa-baseline", action="store_true")
    parser.add_argument("--bobot-tl", type=Path)
    parser.add_argument("--simulasi", action="store_true")
    parser.add_argument("--_simulasi-terisolasi", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--final", action="store_true", help="Evaluasi uji pada sprint terakhir.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Gunakan seed berbeda dalam rentang [0, 2³²).")
    if args.simulasi:
        if args.final:
            parser.error("--final membutuhkan data primer.")
        reproduksi_simulasi(args)
        return

    try:
        data_info = informasi_data()
        hash_data = sidik_data()
    except FileNotFoundError as exc:
        parser.exit(1, f"Data belum siap. Jalankan python -m src.protokol.siapkan. {exc}\n")
    if data_info["jenis"] != ("simulasi" if args._simulasi_terisolasi else "primer"):
        parser.error("Gunakan --simulasi untuk data sintetis atau siapkan data primer.")
    if args.final and args._simulasi_terisolasi:
        parser.error("--final membutuhkan data primer.")

    baris = []
    if not args.tanpa_baseline:
        baris.append(("Starter", baseline_starter(args.seeds[0], hash_data)))
        for seed in args.seeds:
            perintah = ["-m", "src.pembanding.run", "--seed", str(seed)]
            if args.final and seed == args.seeds[-1]:
                perintah.append("--final")
            jalankan(perintah)
            baris.append(("Pembanding", baca_hasil("baseline", "rf_hog_color", seed, hash_data)))

    for nim in dict.fromkeys(args.hanya or ANGGOTA):
        for seed in args.seeds:
            perintah = [f"models/{nim}/run.py", "--seed", str(seed)]
            if args._simulasi_terisolasi:
                perintah.extend(["--epochs", "1"])
            if ANGGOTA[nim] == "tl":
                if args.bobot_tl:
                    perintah.extend(["--bobot", str(args.bobot_tl.resolve())])
                elif args._simulasi_terisolasi:
                    perintah.append("--bobot-acak-simulasi")
            if args.final and seed == args.seeds[-1]:
                perintah.append("--final")
            jalankan(perintah)
            baris.append((nim, baca_hasil(nim, ANGGOTA[nim], seed, hash_data)))
    tabel(baris, data_info, hash_data)


if __name__ == "__main__":
    main()
