"""Buat citra dan log kondisi sintetis untuk menguji mekanisme Sprint 1."""

import csv
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from src.common.manifest import AKAR
from src.protokol.siapkan import GRUP, buat_dataset


def buat_simulasi():
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory(prefix="kondisi-okupansi-") as folder:
        akar = Path(folder) / "simulasi"
        kamera = akar / "kamera"
        kamera.mkdir(parents=True)
        label, indeks = akar / "label.csv", akar / "kamera_index.csv"
        konfigurasi = ("0 orang", "1 orang duduk", "1 orang berdiri", "2 orang", "3 orang")
        with label.open("w", newline="", encoding="utf-8") as fl, indeks.open(
            "w", newline="", encoding="utf-8"
        ) as fi:
            kondisi, citra = csv.writer(fl), csv.writer(fi)
            kondisi.writerow(["no", "kelompok", "orang", "lampu", "t_mulai_host", "t_selesai_host"])
            citra.writerow(["berkas", "t_host"])
            no = 0
            for g, grup in enumerate(GRUP):
                kombinasi = [(orang, terang) for _ in range(4)
                             for orang in konfigurasi for terang in (0, 255)]
                for i in rng.permutation(len(kombinasi)):
                    orang, terang = kombinasi[i]
                    no += 1
                    awal = 1_800_000_000 + 5 * no
                    kondisi.writerow([no, grup, orang, terang, awal, awal + 4])
                    for j, waktu in enumerate((awal + 0.5, awal + 2, awal + 2.5)):
                        latar = int(35 + 45 * (terang > 0) + 8 * g)
                        noise = rng.integers(0, 25, (112, 112, 3), dtype=np.uint8)
                        img = Image.fromarray(noise + latar)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle((0, 90, 111, 111), fill=(latar + 22,) * 3)
                        jumlah = int(orang[0])
                        for k in range(jumlah):
                            x = (k + 1) * 112 // (jumlah + 1) + int(rng.integers(-5, 6))
                            y = 61 if orang.endswith("duduk") else 44
                            warna = (155 + g * 15, 111 + k * 22, 126)
                            draw.ellipse((x - 7, y - 8, x + 7, y + 6), fill=warna)
                            draw.rectangle((x - 10, y + 6, x + 10, 89), fill=warna)
                        nama = f"f{no:03d}_{j}.jpg"
                        img.save(kamera / nama, quality=90)
                        citra.writerow([nama, waktu])
        buat_dataset([(label, indeks)], simulasi=True)


if __name__ == "__main__":
    if not (AKAR / ".simulasi-sprint").is_file():
        raise SystemExit("Mulai simulasi dengan python run_all.py --simulasi.")
    buat_simulasi()
