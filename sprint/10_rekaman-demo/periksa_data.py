"""Tampilkan kontrak data dan split untuk demo; tidak menilai model pada data uji."""

import json
from pathlib import Path
import sys

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.common import manifest, split
from src.protokol import data


def main():
    asal = json.loads((AKAR / "data/okupansi/provenance.json").read_text("utf-8"))
    arrays = data.muat()
    print("Jenis data:", asal["jenis"])
    print("Data hash:", manifest.data_hash(f"split:{split.SEED_SPLIT}".encode()))
    kelompok = []
    for nama, x, y in zip(("train", "val", "test"), arrays[::2], arrays[1::2]):
        assert x.dtype == np.uint8 and x.shape == (len(y), 224, 224, 3)
        assert np.issubdtype(y.dtype, np.integer) and np.isin(y, range(4)).all()
        grup = asal["kelompok_split"][nama]
        kelompok.append(set(grup))
        print(f"{nama}: {len(y)} citra | kelompok {', '.join(grup)} | RGB uint8 224x224")
    assert all(not kelompok[i] & kelompok[j] for i in range(3) for j in range(i + 1, 3))
    print("ID kelompok terpisah antar-split. Label target: 0, 1, 2, 3.")
    print("Pemeriksaan ini tidak melakukan evaluasi model pada data uji.")


if __name__ == "__main__":
    main()
