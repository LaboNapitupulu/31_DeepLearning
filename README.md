# Tim 31 · Estimasi okupansi area meja

Proyek ini mengestimasi jumlah orang di area meja dari citra RGB 224×224 piksel. Targetnya empat kelas, 0 hingga 3 orang. Posisi duduk dan berdiri dikelompokkan dalam kelas 1.

Data primer disesuaikan menjadi **160 citra**. Pemisahan berdasarkan kelompok menghasilkan 80 citra latih (G2 dan G3), 40 validasi (G1), dan 40 uji (G4). Hasil yang tercatat pada Sprint 1 berasal dari data validasi dengan seed 1.

## Hasil validasi

| Model | Akurasi | Macro-F1 | MAE (orang) |
|---|---:|---:|---:|
| Baseline starter | 25,00% | 0,1000 | 1,2250 |
| Random Forest + HOG + histogram warna | 60,00% | 0,5413 | 0,6000 |
| MobileNetV2 | 65,00% | 0,4318 | 0,4250 |
| CNN | 47,50% | 0,3796 | 0,7250 |
| Transformer | 42,50% | 0,3108 | 0,7250 |
| GRU | 57,50% | 0,4157 | 0,6250 |

Angka validasi mengikuti artefak primer terbaru di `results/` dan
[laporan komparasi](reports/hasil_komparasi.md). Run lokal CNN memiliki satu
prediksi berbeda dari artefak awal tim; perbedaan tersebut dicatat tanpa
mengubah hasil agar mengikuti laporan lama. Bukti uji integrasi awal tetap
tersedia di [tabel uji terpisah](sprint/10_rekaman-demo/hasil-komparasi-lokal.md).

## Demo Sprint 1

[Tonton video Demo Sprint 1 Tim 31 di Google Drive](https://drive.google.com/file/d/1SSENODW6TGfoZSxkIcr_du-m5E-u_YLp/view?usp=sharing).

Penanggung jawab rekaman: Labo Napitupulu (123450037).
[Dokumentasi tugas #10](sprint/10_rekaman-demo/README.md) memuat tautan video,
bukti uji runner, dan panduan reproduksi. Video di Drive menjadi rujukan
pengumpulan; file video tidak disalin ke Git.

## Menjalankan ulang di Windows

Jalankan PowerShell dari akar repositori. Lingkungan `.venv` di laptop Labo
sudah memuat dependensi proyek. Untuk lingkungan Python baru, buat `.venv`
dengan Python 3.12, pasang PyTorch/torchvision CPU, lalu dependensi lainnya:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sebelum menjalankan model, pastikan tersedia:

- `data/okupansi/dataset.npz` dan `data/okupansi/provenance.json` dari paket data tim.
- `data/MANIFEST.json` dari repositori. Hash primer yang dipakai Sprint 1:
  `ae9811140ba30965a9d7c5e94fabd85d`.
- `data/mobilenet_v2-7ebf99e0.pth`, bobot ImageNet MobileNetV2.

Data dan bobot tidak disimpan di Git. Paket `DAATAA.zip` yang diterima tim
menyediakan dataset, citra sumber, anotasi visual, dan indeks kamera. Bobot
MobileNetV2 dapat diunduh dari sumber PyTorch:

```powershell
Invoke-WebRequest https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth -OutFile data/mobilenet_v2-7ebf99e0.pth
```

Untuk demo Sprint 1, periksa data, lalu latih seluruh model dengan satu seed:

```powershell
.\.venv\Scripts\python.exe -B sprint/10_rekaman-demo/periksa_data.py
.\.venv\Scripts\python.exe -u -B run_all.py --seeds 1 --bobot-tl data/mobilenet_v2-7ebf99e0.pth
```

Runner menjalankan baseline starter, Histogram RGB + HOG + Random Forest,
MobileNetV2, CNN, Transformer, dan GRU. Hasil terbaru ditulis ke `results/`
dan tabel `reports/hasil_komparasi.md`, termasuk durasi run tersebut.
Tanpa `--final`, hanya prediksi validasi yang dinilai. Evaluasi uji G4
ditunda sampai sprint terakhir oleh Penjaga Data.

Untuk uji mekanisme dengan citra sintetis:

```powershell
.\.venv\Scripts\python.exe -u -B run_all.py --simulasi --seeds 1
```

Hasil sintetis disimpan terpisah di `results/simulasi/` dan tidak dipakai
sebagai angka komparasi data primer. [Panduan merekam demo #10](sprint/10_rekaman-demo/panduan-rekaman.md)
memuat susunan video dan naskah singkat. [Bukti uji integrasi lokal](sprint/10_rekaman-demo/hasil-integrasi-lokal.json)
memuat hasil verifikasi runner pada salinan kode dan data yang terpisah.

## Berkas proyek

| Berkas | Isi |
|---|---|
| [run_all.py](run_all.py) | Runner simulasi dan data primer |
| [Laporan komparasi](reports/hasil_komparasi.md) | Metrik validasi seed 1, durasi, dan data hash |
| [Indeks model](models/README.md) | Kode, metrik, dan prediksi per model |
| [Indeks sprint](sprint/README.md) | Berkas per kartu Workbench |
