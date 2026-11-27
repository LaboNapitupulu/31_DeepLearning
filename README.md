# Tim 31 · Estimasi okupansi area meja

Proyek ini mengestimasi jumlah orang di area meja dari citra RGB 224×224 piksel. Targetnya empat kelas, 0 hingga 3 orang. Posisi duduk dan berdiri dikelompokkan dalam kelas 1.

Data primer disesuaikan menjadi **160 citra**. Pemisahan berdasarkan kelompok menghasilkan 80 citra latih (G2 dan G3), 40 validasi (G1), dan 40 uji (G4). Hasil yang tercatat pada Sprint 1 berasal dari data validasi dengan seed 1.

## Hasil validasi

| Model | Akurasi | Macro-F1 | MAE (orang) |
|---|---:|---:|---:|
| Baseline starter | 25,00% | 0,1000 | 1,2250 |
| Random Forest + HOG + histogram warna | 60,00% | 0,5413 | 0,6000 |
| MobileNetV2 | 65,00% | 0,4318 | 0,4250 |
| CNN | 47,50% | 0,3907 | 0,7000 |
| Transformer | 42,50% | 0,3108 | 0,7250 |
| GRU | 57,50% | 0,4157 | 0,6250 |

Angka validasi diambil dari run seed 1 yang tercatat di laporan komparasi.

## Berkas proyek

| Berkas | Isi |
|---|---|
| [run_all.py](run_all.py) | Runner simulasi dan data primer |
| [Laporan komparasi](reports/hasil_komparasi.md) | Metrik validasi seed 1, durasi, dan data hash |
| [Indeks model](models/README.md) | Kode, metrik, dan prediksi per model |
| [Indeks sprint](sprint/README.md) | Berkas per kartu Workbench |
