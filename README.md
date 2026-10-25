# Capstone Deep Learning — starter tim

Template repositori tim untuk **Capstone Deep Learning: Satu Kasus, Lima Keluarga Model**
(Program Studi Sains Data ITERA, Data Science Workbench). Satu tim = satu repositori yang
dibuat dari template ini; setiap anggota memiliki satu keluarga model dan satu folder.

Kasus: klasifikasi citra **Fashion-MNIST** 10 kelas. Data sudah ada di laptop Anda
(`~/workbench-workspace/deep-learning/data/raw/FashionMNIST/raw/`, disiapkan Workbench pada
Modul 03–06). Protokol evaluasi dibekukan dosen di `src/common/` dan **tidak boleh diubah**.

## Struktur

```text
src/common/        milik dosen — jangan diubah (data, split beku, evaluate, baseline)
models/<nim>/      milik satu anggota — kode model, run.py
results/<nim>/     berkas hasil anggota: pred_val.npy, pred_test.npy, metrics.json
reports/individu/<nim>.md   laporan perorangan (≤ 1200 kata)
reports/tim.md     laporan tim (6 bagian)
sprint/<sprint>/<nim>.md    catatan sprint (opsional; Log Sprint resmi ada di Workbench)
AI_LOG.md          catatan pemakaian AI tim
run_all.py         satu perintah: jalankan semua model dan rakit tabel komparasi
```

## Mulai

1. Satu anggota (Integrator) membuat repo dari template ini (**Use this template**), lalu
   mengundang semua anggota sebagai kolaborator dan mengikat repo di Papan Tim Workbench.
2. Setiap anggota membuka folder tim dari aplikasi DSWorkbench (panel Capstone); aplikasi
   membuat cabang `<nim>/kerja`. Bekerja hanya di `models/<nim>/`, `results/<nim>/`,
   `reports/individu/<nim>.md`, dan `sprint/*/<nim>.md`.
3. Jalankan baseline dosen sekali untuk memastikan data dan protokol jalan:

   ```bash
   pip install -r requirements.txt
   python -m src.common.baseline_mlp --seed 2026
   ```

   Hasilnya di `results/baseline/metrics.json`. Angka baseline adalah batas bawah yang harus
   dilampaui setiap model.
4. Salin `models/TEMPLATE/` menjadi `models/<nim>/`, isi `run.py` sesuai kontrak di
   `models/README.md`, lalu jalankan:

   ```bash
   python models/<nim>/run.py --seed 1 && python models/<nim>/run.py --seed 2 && python models/<nim>/run.py --seed 3
   ```

   Setiap run menulis `results/<nim>/metrics.json` lewat `src.common.evaluate`; Local Runner
   memeriksanya (skema `metrik/seed/data_hash`, isi sama dengan HEAD) saat Anda menekan
   *Periksa artefak* di aplikasi.
5. Perintah reproduksi tim (dinilai pada sprint terakhir):

   ```bash
   python run_all.py
   ```

## Protokol yang dibekukan

- Split latih/validasi dari 60.000 data latih resmi: permutasi dengan seed 2026, 6.000
  pertama = validasi, sisanya latih (`src/common/split.py`).
- Data uji resmi (10.000) hanya dievaluasi **sekali** oleh Penjaga Data pada sprint
  terakhir (`--pred-test`). Evaluasi berulang pada data uji dianggap kebocoran.
- Metrik utama: macro-F1 dan akurasi pada validasi; dilaporkan sebagai rerata ± simpangan
  baku dari 3 seed pada konfigurasi akhir.
- Anggaran: ≤ 20 menit pelatihan per run di CPU laptop. Catat `--durasi` dan `--perangkat`.
- `data_hash` dihitung `evaluate.py` dari berkas label dan split; jangan diisi manual.

## Jalur model

| Jalur | Keluarga | Modul |
|---|---|---|
| `cnn` | CNN dari nol | 05 |
| `tl` | Transfer learning ResNet18 (`data/raw/torch`) | 06 |
| `rnn` | RNN → LSTM/GRU atas baris citra (28 langkah × 28 fitur) | 07–08 |
| `trf` | Transformer encoder atas patch | 09 |
| `ae` | Autoencoder → klasifier linear / anomali | — |

Syarat semua model: optimizer dengan alasan tertulis, ≥ 1 regularisasi, tuning
hyperparameter dengan catatan pencarian, ≥ 2 ablation, 3 seed pada konfigurasi akhir.

Lisensi: MIT. Repo ini masuk Galeri Capstone sebagai portofolio setelah semester berakhir.
