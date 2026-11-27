"""Cocokkan citra dan kondisi, lalu bekukan data okupansi yang dipakai semua jalur."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps

from src.common import manifest, split

FOLDER = manifest.AKAR / "data" / "okupansi"
GRUP = ("G1", "G2", "G3", "G4")
BENTUK = (224, 224, 3)
LABEL = {
    "0": 0, "0 orang": 0, "1": 1, "1 orang": 1, "1 orang duduk": 1,
    "1 orang berdiri": 1, "2": 2, "2 orang": 2, "3": 3, "3 orang": 3,
}


def baca_csv(path, kolom):
    with path.open(encoding="utf-8-sig", newline="") as f:
        pembaca = csv.DictReader(f)
        if not set(kolom).issubset(pembaca.fieldnames or ()):
            raise ValueError(f"{path.name}: wajib memiliki kolom {', '.join(kolom)}.")
        return list(pembaca)


def sidik_file(path):
    hash_file = hashlib.sha256()
    with path.open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            hash_file.update(blok)
    return hash_file.hexdigest()


def muat_citra(folder, nama, berkas_dipakai):
    relatif = Path(nama)
    if relatif.is_absolute() or ".." in relatif.parts or not relatif.parts:
        raise ValueError(f"Jalur citra tidak sah: {nama!r}.")
    path = folder / relatif
    if not path.is_file():
        path = folder / "kamera" / relatif
    identitas = path.resolve()
    if identitas in berkas_dipakai:
        raise ValueError(f"Citra yang sama masuk lebih dari sekali: {path}.")
    berkas_dipakai.add(identitas)
    with Image.open(path) as gambar:
        rgb = ImageOps.exif_transpose(gambar).convert("RGB")
        citra = np.asarray(rgb.resize(BENTUK[:2], Image.Resampling.BILINEAR))
    return citra, sidik_file(path)


def olah_pasangan(label_csv, indeks_csv, pengecualian, berkas_dipakai):
    kondisi = baca_csv(label_csv, ("no", "kelompok", "orang", "t_mulai_host", "t_selesai_host"))
    kondisi.sort(key=lambda baris: float(baris["t_mulai_host"]))
    indeks = baca_csv(indeks_csv, ("berkas", "t_host"))
    waktu = np.array([float(baris["t_host"]) for baris in indeks])
    if not np.isfinite(waktu).all() or np.any(np.diff(waktu) <= 0):
        raise ValueError(f"{indeks_csv}: t_host harus terurut naik dan unik.")
    nomor = [baris["no"] for baris in kondisi]
    if len(set(nomor)) != len(nomor) or pengecualian.keys() - set(nomor):
        raise ValueError(f"{label_csv}: nomor kondisi ganda atau pengecualian tidak ditemukan.")

    citra, label, metadata = [], [], []
    kondisi_kosong, dipakai = [], Counter()
    akhir_lalu = -np.inf
    for baris in kondisi:
        awal, akhir = float(baris["t_mulai_host"]), float(baris["t_selesai_host"])
        if not np.isfinite([awal, akhir]).all() or awal < akhir_lalu or akhir <= awal:
            raise ValueError(f"{label_csv}: selang kondisi {baris['no']} bertumpuk atau tidak sah.")
        akhir_lalu = akhir
        grup = baris["kelompok"].strip()
        kelas = LABEL.get(baris["orang"].strip().lower())
        if grup not in GRUP or kelas is None:
            raise ValueError(f"{label_csv}: kelompok atau label orang tidak sah pada kondisi {baris['no']}.")
        if baris["no"] in pengecualian:
            continue
        pertama = np.searchsorted(waktu, awal + 1)
        batas = np.searchsorted(waktu, akhir)
        if pertama >= batas:
            kondisi_kosong.append(baris["no"])
            continue
        dipakai[grup] += 1
        for foto in indeks[pertama:batas]:
            gambar, sha256 = muat_citra(indeks_csv.parent, foto["berkas"], berkas_dipakai)
            citra.append(gambar)
            label.append(kelas)
            metadata.append({
                "kelompok": grup, "kondisi": baris["no"], "berkas": Path(foto["berkas"]).as_posix(),
                "t_host": float(foto["t_host"]), "orang": baris["orang"],
                "lampu": baris.get("lampu"), "sha256": sha256,
            })

    ringkasan = {
        "sesi": label_csv.parent.name, "kamera": indeks_csv.parent.name,
        "sha256_label": sidik_file(label_csv), "sha256_indeks": sidik_file(indeks_csv),
        "jumlah_kondisi": len(kondisi), "kondisi_dipakai_per_kelompok": dict(dipakai),
        "kelompok_direncanakan": dict(Counter(b["kelompok"] for b in kondisi)),
        "kondisi_tanpa_citra": kondisi_kosong, "kondisi_dikeluarkan": pengecualian,
        "jumlah_citra_dipakai": len(label),
        "median_interval_kamera_detik": round(float(np.median(np.diff(waktu))), 4)
        if len(waktu) > 1 else None,
    }
    return citra, label, metadata, ringkasan


def olah_anotasi(anotasi_csv, indeks_csv, berkas_dipakai):
    anotasi = baca_csv(anotasi_csv, (
        "berkas", "berkas_asli", "t_host", "okupansi", "kelompok_blok_sumber",
        "kelompok_rancangan", "no_slot_rancangan",
    ))
    baris_indeks = baca_csv(indeks_csv, ("berkas", "t_host"))
    waktu = np.array([float(r["t_host"]) for r in baris_indeks])
    indeks = {r["berkas"]: t for r, t in zip(baris_indeks, waktu)}
    if (len(indeks) != len(baris_indeks) or not np.isfinite(waktu).all()
            or np.any(np.diff(waktu) <= 0)):
        raise ValueError(f"{indeks_csv}: nama harus unik dan t_host terurut naik.")
    nama = {r["berkas"] for r in anotasi}
    sumber = {r["berkas_asli"] for r in anotasi}
    slot = {r["no_slot_rancangan"] for r in anotasi}
    if (not anotasi or nama != indeks.keys() or len(nama) != len(anotasi)
            or len(sumber) != len(anotasi) or "" in sumber
            or len(slot) != len(anotasi) or "" in slot):
        raise ValueError(f"{anotasi_csv}: indeks, nama JPEG, sumber, dan slot harus berpasangan satu-ke-satu.")

    citra, label, metadata = [], [], []
    for row in anotasi:
        file = row["berkas"]
        kelas = LABEL.get(row["okupansi"])
        grup_sumber = row["kelompok_blok_sumber"]
        grup = row["kelompok_rancangan"]
        if (kelas is None or grup not in GRUP or (grup_sumber and grup_sumber != grup)
                or float(row["t_host"]) != indeks[file]):
            raise ValueError(f"{anotasi_csv}: label, kelompok, atau waktu tidak cocok untuk {file}.")
        gambar, sha256 = muat_citra(indeks_csv.parent, file, berkas_dipakai)
        citra.append(gambar)
        label.append(kelas)
        metadata.append({
            "berkas": file, "berkas_asli": row["berkas_asli"], "t_host": float(indeks[file]),
            "kelompok": grup, "kelompok_blok_sumber": grup_sumber,
            "kondisi": row["no_slot_rancangan"],
            "kondisi_saat_foto": row.get("no_kondisi_saat_foto"), "orang": kelas,
            "orang_rancangan": row.get("orang_rancangan"),
            "lampu_rancangan": row.get("lampu_rancangan"), "sha256": sha256,
        })
    ringkasan = {
        "kamera": indeks_csv.parent.name,
        "sha256_anotasi": sidik_file(anotasi_csv),
        "sha256_indeks": sidik_file(indeks_csv),
        "jumlah_citra_dipakai": len(label),
        "jumlah_citra_luar_blok": sum(not r["kelompok_blok_sumber"] for r in metadata),
        "kondisi_dipakai_per_kelompok": dict(Counter(r["kelompok"] for r in metadata)),
    }
    return citra, label, metadata, ringkasan


def buat_dataset(pasangan, dikeluarkan=None, simulasi=False, anotasi_visual=None):
    citra, label, metadata, sesi = [], [], [], []
    berkas_dipakai = set()
    dikeluarkan = dikeluarkan or {}
    if anotasi_visual is not None:
        x, y, m, ringkasan = olah_anotasi(*anotasi_visual, berkas_dipakai)
        citra.extend(x)
        label.extend(y)
        metadata.extend({"pasangan": 1, **baris} for baris in m)
        sesi.append(ringkasan)
    for n, (label_csv, indeks_csv) in enumerate(pasangan, 1):
        x, y, m, ringkasan = olah_pasangan(
            label_csv, indeks_csv, dikeluarkan.get(n, {}), berkas_dipakai
        )
        citra.extend(x)
        label.extend(y)
        for baris in m:
            metadata.append({"pasangan": n, **baris})
        sesi.append(ringkasan)
        print(f"Pasangan {n}: {len(y)} citra, kondisi terpakai per kelompok "
              f"{ringkasan['kondisi_dipakai_per_kelompok']}.", flush=True)

    if not metadata:
        raise ValueError("Tidak ada citra dalam jendela kondisi. Periksa pasangan sesi dan waktu indeks.")
    grup = np.array([baris["kelompok"] for baris in metadata])
    y = np.array(label, dtype=np.int64)
    distribusi = {g: dict(sorted(Counter(y[grup == g].tolist()).items())) for g in GRUP}
    if any(set(dist) != set(range(4)) for dist in distribusi.values()):
        raise ValueError(f"Empat kelompok wajib mencakup kelas 0–3. Distribusi aktual: {distribusi}")

    sisa, uji = split.indeks(len(GRUP), porsi_val=0.25)
    latih, val = split.indeks(len(sisa), porsi_val=1 / 3)
    daftar_grup = np.array(GRUP)
    grup_split = {"train": daftar_grup[sisa[latih]], "val": daftar_grup[sisa[val]],
                  "test": daftar_grup[uji]}
    data_split = {nama: np.flatnonzero(np.isin(grup, kelompok)) for nama, kelompok in grup_split.items()}
    asal = {
        "jenis": "simulasi" if simulasi else "primer",
        **({"sumber_label": "anotasi_visual"} if anotasi_visual is not None else {}),
        "seed_split": split.SEED_SPLIT,
        "kelompok_split": {nama: kelompok.tolist() for nama, kelompok in grup_split.items()},
        "ukuran_split": {nama: len(idx) for nama, idx in data_split.items()},
        "distribusi_kelompok": distribusi, "sesi": sesi, "citra": metadata,
    }
    FOLDER.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=FOLDER, prefix=".siapkan-") as sementara:
        d = Path(sementara)
        np.savez_compressed(d / "dataset.npz", X=np.stack(citra), y=y, **data_split)
        (d / "provenance.json").write_text(json.dumps(asal, indent=2) + "\n", encoding="utf-8")
        for nama in ("dataset.npz", "provenance.json"):
            (d / nama).replace(FOLDER / nama)
    print(f"{asal['jenis']}: {asal['ukuran_split']} | kelompok {asal['kelompok_split']}", flush=True)
    print("data_hash:", manifest.data_hash(f"split:{split.SEED_SPLIT}".encode()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sumber = parser.add_mutually_exclusive_group(required=True)
    sumber.add_argument("--pasangan", nargs=2, action="append", type=Path, default=[],
                        metavar=("LABEL_CSV", "INDEKS_KAMERA"))
    sumber.add_argument("--anotasi-visual", nargs=2, type=Path,
                        metavar=("ANOTASI_CSV", "INDEKS_KAMERA"))
    parser.add_argument("--keluarkan", nargs=2, action="append", default=[],
                        metavar=("PASANGAN:NO_KONDISI", "ALASAN"))
    args = parser.parse_args()
    if args.anotasi_visual and args.keluarkan:
        parser.error("--keluarkan hanya berlaku untuk --pasangan.")
    pengecualian = {}
    for kunci, alasan in args.keluarkan:
        bagian, _, kondisi = kunci.partition(":")
        if (not bagian.isdecimal() or not 1 <= int(bagian) <= len(args.pasangan)
                or not kondisi or not alasan.strip()):
            parser.error("--keluarkan memerlukan nomor pasangan, nomor kondisi, dan alasan.")
        pengecualian.setdefault(int(bagian), {})[kondisi] = alasan
    try:
        buat_dataset(args.pasangan, pengecualian, anotasi_visual=args.anotasi_visual)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Persiapan data gagal: {exc}\n")


if __name__ == "__main__":
    main()
