# 02 · Data dan protokol

Berkas untuk ditinjau: `.gitignore`, `requirements.txt`, `data/MANIFEST.json`,
`data/CATATAN-SESI.md`, `src/protokol/`, `src/eksperimen.py`,
`src/latih_citra.py`, dan `sprint/02_data-dan-protokol/README.md`.

Untuk membangun ulang dataset Tim 31 yang telah disesuaikan menjadi tepat
160 JPEG, gunakan anotasi visual dan indeks dari direktori kamera yang sama.

```bash
python -m src.protokol.siapkan --anotasi-visual \
  data/tim31_web-20261123-120524-ed41/anotasi_visual.csv \
  data/tim31_web-20261123-120524-ed41/kamera_index.csv
```

Perintah ini mencocokkan nama JPEG dan waktu pada kedua CSV, lalu memakai
kolom `okupansi` sebagai label. Pembagian per kelompok menghasilkan 80 citra
latih (G2/G3), 40 validasi (G1), dan 40 uji (G4). Delapan belas foto di luar
blok mengikuti kelompok alokasinya. Log asli tetap menjadi catatan rancangan.

Untuk mengulang pemrosesan versi awal dari rekaman mentah, ekstrak TAR kamera
ke lokasi terpisah dan gunakan indeks serta JPEG dari ekstraksi tersebut.
Ulangi `--pasangan` bila memakai beberapa sesi.

```bash
python -m src.protokol.siapkan \
  --pasangan data/tim31_sesi-20261123-120519/sesi-20261123-120519/label.csv \
  /path/ke/ekstraksi-rekaman-mentah/kamera_index.csv
```

Untuk kondisi yang terganggu, tambahkan misalnya `--keluarkan 2:19 "Partisipan masih berpindah"`.
Persiapan mencetak jumlah kondisi yang terpakai per kelompok. Empat kelompok
wajib memiliki sampel untuk kelas 0–3. Bagian latih adalah G2/G3, validasi G1, dan uji G4.
Baca temuan pemeriksaan arsip yang tersedia di `data/CATATAN-SESI.md`.

Bagikan dataset turunan `data/okupansi/` melalui penyimpanan internal tim agar semua
anggota memakai byte dan hash yang sama. Lampirkan rekap mutu ke PR protokol.
