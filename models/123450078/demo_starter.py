"""Buktikan baseline starter berjalan dengan contoh Fashion-MNIST dari dosen.

    python3 models/123450078/demo_starter.py --seed 2026 --publish-demo

Protokol Fashion-MNIST dijalankan pada salinan sementara di /tmp.
Hasil demo: results/123450078/demo_starter/, terpisah dari hasil kasus tim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

AKAR = Path(__file__).resolve().parents[2]
NIM = Path(__file__).resolve().parent.name
BERKAS_IDX = (
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--publish-demo", action="store_true",
        help="ganti artefak NIM yang masih placeholder dengan hasil berlabel demo",
    )
    ap.add_argument(
        "--data-dir", type=Path,
        default=AKAR.parent / "deep-learning/data/raw/FashionMNIST/raw",
        help="lokasi empat berkas IDX Fashion-MNIST, mentah atau .gz",
    )
    a = ap.parse_args()
    if a.threads < 1:
        ap.error("threads harus positif")
    artefak = AKAR / "results" / NIM / "metrics.json"
    if a.publish_demo and artefak.exists():
        sebelumnya = json.loads(artefak.read_text("utf-8"))
        if sebelumnya and sebelumnya.get("status") not in {"belum_dievaluasi", "demo_starter_berhasil"}:
            ap.error("artefak NIM sudah berisi hasil selain placeholder/demo; tidak ditimpa")
    lokasi = a.data_dir.expanduser().resolve()
    berkas = []
    for nama in BERKAS_IDX:
        p = lokasi / nama
        if not p.is_file():
            p = lokasi / (nama + ".gz")
        if not p.is_file():
            ap.error(f"berkas dataset belum tersedia: {lokasi / nama} (atau .gz)")
        berkas.append(p)

    env = os.environ.copy()
    for nama in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[nama] = str(a.threads)
    # Hindari PYTHONPATH laptop mengarahkan evaluator ke protokol asli repo tim.
    env.pop("PYTHONPATH", None)
    perintah = [sys.executable, "-m", "src.common.baseline", "--seed", str(a.seed)]
    t0 = time.perf_counter()
    print("DEMO Fashion-MNIST: baseline dan evaluator bawaan starter.", flush=True)
    print("Protokol kasus penghitung orang tetap berada di repo utama.", flush=True)
    with tempfile.TemporaryDirectory(prefix="capstone-demo-", dir="/tmp") as direktori:
        demo = Path(direktori)
        shutil.copytree(AKAR / "src", demo / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(
            AKAR / "src/protokol/contoh_fashion_mnist.py", demo / "src/protokol/data.py",
        )
        raw = demo / "data/FashionMNIST/raw"
        raw.mkdir(parents=True)
        for p in berkas:
            (raw / p.name).symlink_to(p)
        manifes = {
            "sumber": "Fashion-MNIST lokal, contoh dataset pada starter dosen",
            "berkas": [f"data/FashionMNIST/raw/{p.name}" for p in berkas],
            "catatan": "Demo baseline starter; bukan dataset penghitung orang di meja.",
        }
        (demo / "data/MANIFEST.json").write_text(
            json.dumps(manifes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        selesai = subprocess.run(perintah, cwd=demo, env=env, capture_output=True, text=True)
        print(selesai.stdout, end="", flush=True)
        if selesai.stderr:
            print(selesai.stderr, end="", file=sys.stderr, flush=True)
        if selesai.returncode:
            raise SystemExit(selesai.returncode)
        hasil = demo / "results/baseline"
        metrics = json.loads((hasil / "metrics.json").read_text("utf-8"))
        out = AKAR / "results" / NIM / "demo_starter"
        out.mkdir(parents=True, exist_ok=True)
        # Pertahankan metrics persis sebagaimana ditulis evaluator dosen.
        for nama in ("metrics.json", "pred_val.npy"):
            shutil.copyfile(hasil / nama, out / nama)
        shutil.copyfile(demo / "data/MANIFEST.json", out / "MANIFEST.json")
        (out / "run.log").write_text(selesai.stdout + selesai.stderr, encoding="utf-8")
    bukti = {
        "status": "demo_starter_berhasil",
        "dataset": "Fashion-MNIST",
        "tujuan": "Menguji baseline, split, manifest, dan evaluator starter dosen",
        "batasan": "Bukan pelatihan Transformer atau evaluasi kasus penghitung orang tim",
        "nim": NIM,
        "seed": a.seed,
        "threads": a.threads,
        "perintah_reproduksi": (
            f"python3 models/{NIM}/demo_starter.py --seed {a.seed} --threads {a.threads}"
            + (" --publish-demo" if a.publish_demo else "")
        ),
        "data_dir": str(lokasi),
        "durasi_total_detik": round(time.perf_counter() - t0, 3),
        "data_uji_dievaluasi": False,
    }
    (out / "demo.json").write_text(
        json.dumps(bukti, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Bukti demo starter dosen\n\n"
        "Dataset: Fashion-MNIST lokal dengan contoh pemuat bawaan starter.\n"
        "Model: baseline regresi logistik NumPy dari starter.\n"
        "Hasil ini untuk membuktikan alur starter berjalan; kasus tim tetap penghitung orang.\n\n"
        f"- Seed: {metrics['seed']}\n"
        f"- Ukuran validasi: {metrics['n_val']}\n"
        f"- Akurasi validasi: {metrics['metrik']['val_acc']:.6f}\n"
        f"- Macro-F1 validasi: {metrics['metrik']['val_macro_f1']:.6f}\n"
        f"- Durasi baseline (detik): {metrics['durasi_detik']}\n"
        f"- Data hash: `{metrics['data_hash']}`\n"
        "- Data uji tidak dievaluasi.\n\n"
        "`metrics.json` disalin tanpa perubahan dari evaluator starter; `pred_val.npy`\n"
        "berisi prediksi asli. Detail reproduksi ada di `demo.json` dan log di `run.log`.\n\n"
        "```bash\n" + bukti["perintah_reproduksi"] + "\n```\n\n"
        "Penerimaan demo untuk tahap pengumpulan mengikuti ketentuan dosen.\n",
        encoding="utf-8",
    )
    if a.publish_demo:
        publikasi = {
            **metrics,
            "status": "demo_starter_berhasil",
            "dataset": "Fashion-MNIST",
            "tujuan": "Bukti alur starter berjalan; bukan evaluasi kasus penghitung orang tim",
            "bukti_demo": str(out.relative_to(AKAR)),
        }
        shutil.copyfile(out / "pred_val.npy", artefak.parent / "pred_val.npy")
        artefak.write_text(
            json.dumps(publikasi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(f"Artefak NIM berlabel demo: {artefak.relative_to(AKAR)}", flush=True)
    print(f"Bukti demo tersimpan di {out.relative_to(AKAR)}", flush=True)


if __name__ == "__main__":
    main()
