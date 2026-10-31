# Transformer — 123450078

## Status

Kode awal tersedia; belum dilatih karena rekaman primer tim belum ada.
`results/123450078/metrics.json` baru dibuat setelah pelatihan dan evaluasi berhasil.
Belum ada angka hasil eksperimen, tuning, ablation, ukuran ONNX, atau latensi.

## Kasus dari kartu tim

Klasifikasi jumlah orang di depan meja: kelas 0, 1, 2, 3; citra 224×224.
Kondisi satu orang duduk dan satu orang berdiri sama-sama berlabel 1.
PIR dicatat sebagai metadata pengambilan/label sesuai keputusan tim;
masukan Transformer hanya citra.

Rancangan rekaman: G1–G4, lima kondisi orang, lampu 128/255,
empat ulangan, empat detik per kondisi, kamera setiap 0,5 detik.
Buang citra dengan waktu relatif terhadap awal kondisi kurang dari satu detik.
Data primer direkam sendiri. Data tim lain hanya untuk uji tambahan;
dataset pengganti memerlukan persetujuan pembimbing sesuai kartu kasus.

Operator meja: 10 November 2026 pukul 14.00–15.00.
Uji purwarupa: 18 November 2026 pukul 15.00–15.30.

## Data dan split yang masih harus disiapkan tim

Simpan citra di `data/`. Catat jalur citra, jumlah orang, sesi, blok,
identitas partisipan, kondisi, waktu relatif kondisi, PIR, lampu, dan ulangan
dalam metadata. Jangan memasukkan identitas partisipan atau PIR sebagai fitur model ini.
Isi `data/MANIFEST.json` dengan metadata dan seluruh berkas citra yang benar-benar dipakai;
hash tidak boleh hanya mencakup metadata tanpa citranya.

Penjaga Data mengisi `src/protokol/data.py` dan membekukan `PROTOKOL.md` melalui
PR sesuai aturan tim. Pemuat mengembalikan enam array sesuai kontrak starter.
Untuk skrip anggota ini: X berupa RGB uint8 `[N,224,224,3]` atau `[N,3,224,224]`,
y berupa integer `[N]` dengan nilai 0–3. Citra belum dinormalisasi oleh pemuat;
statistik kanal dihitung skrip dari data latih saja.

Kartu kasus mensyaratkan GroupKFold berdasarkan sesi/blok/partisipan.
Jangan memakai split acak per citra dari template: citra berdekatan dari kelompok
yang sama dapat bocor antarbagian. Definisi kelompok, jumlah fold, fold validasi/uji,
dan pencegahan partisipan yang sama melintasi bagian harus disepakati dan dibekukan
oleh tim sebelum pelatihan. Skrip ini menerima split protokol, tidak membuat split sendiri.

## Arsitektur dan optimizer awal

Patch embedding berukuran 32×32 menghasilkan 49 token berdimensi 64.
Embedding posisi dipelajari; dua lapisan Transformer encoder dengan empat head,
feed-forward 128, mean pooling, LayerNorm, dan classifier empat kelas.
Patch embedding memakai Conv2d dengan stride sama dengan ukuran patch.
Setiap encoder dibuat terpisah agar parameter awalnya independen.

AdamW, learning rate 0,0003, weight decay 0,01, batch 16, sepuluh epoch.
AdamW dipilih sebagai titik awal karena mendukung pembaruan adaptif dan weight decay
terpisah; kecocokannya untuk data meja belum diuji. Regularisasi awal:
dropout 0,1 dan weight decay. Model sengaja kecil sebagai kandidat untuk CPU laptop;
kelayakan ukuran dan latensinya tetap harus diukur.

## Cara menjalankan setelah data dan protokol tersedia

Dari akar repo:

```bash
python3 models/123450078/run.py --seed 1
python3 models/123450078/run.py --seed 2
python3 models/123450078/run.py --seed 3
```

Skrip memanggil `src.common.evaluate` untuk menulis metrics resmi (`model: trf`).
Hasil terakhir disalin ke `results/123450078/metrics.json` untuk pemeriksaan artefak.
Prediksi, metrics, konfigurasi, riwayat loss, dan checkpoint setiap seed disimpan di
`results/123450078/seed_<seed>/`; checkpoint `.pt` diabaikan Git.
Angka akurasi/macro-F1 berasal dari penilai starter. Karena penilai starter belum
menyediakan MAE untuk klasifikasi, MAE jumlah orang ditulis terpisah di
`seed_<seed>/detail.json`, tanpa menyunting penilai maupun metrics resmi.
Durasi yang dicatat mencakup pemuatan data, normalisasi, pelatihan, dan prediksi.

Evaluasi data uji hanya dijalankan dengan `--final`, sekali oleh Penjaga Data
pada sprint terakhir. Jangan memakai hasil uji untuk tuning.

Setelah hasil asli tersedia, commit kode, prediksi, metrics, dan detail hasil lalu
push ke `123450078/kerja`. Pemeriksa Workbench mengharuskan artefak sama dengan HEAD.

## Pencarian hyperparameter (rencana, belum dikerjakan)

Bandingkan learning rate 0,0001 dan 0,0003 pada split validasi yang sama.
Catat konfigurasi, seed, akurasi, MAE, dan durasi setiap percobaan.
Gunakan hanya validasi untuk memilih konfigurasi akhir.

## Ablation (rencana, belum dikerjakan)

1. Matikan dropout dengan `--dropout 0`, pengaturan lainnya sama.
2. Matikan weight decay dengan `--weight-decay 0`, pengaturan lainnya sama.

## Hasil akhir yang masih diperlukan

Laporkan akurasi dan MAE sebagai rerata ± simpangan baku dari tiga seed konfigurasi
akhir. Ekspor model ke ONNX dan ukur ukuran berkas (KB) serta latensi (ms) pada
laptop meja dengan prosedur warm-up, jumlah pengulangan, thread, dan batas
praproses yang sama untuk semua model. Ekspor dan benchmark ONNX belum diimplementasikan.
Bandingkan dengan Histogram warna + HOG + Random Forest pada split identik;
pembanding ini juga belum tersedia di repo.

## Uji kesiapan starter dengan data demo

Baseline dan evaluator starter telah berhasil dijalankan memakai Fashion-MNIST
lokal dengan contoh pemuat dari dosen. Cara mengulangi:

```bash
python3 models/123450078/demo_starter.py --seed 2026 --publish-demo
```

Skrip membuat salinan sementara kode starter dan mengaktifkan protokol contoh
hanya di salinan tersebut. Hasil lengkap, manifest demo, log, dan perintah
reproduksi disimpan di `results/123450078/demo_starter/`.
Dengan `--publish-demo`, artefak `results/123450078/metrics.json` dan
`pred_val.npy` berisi hasil baseline demo; metrics diberi status
`demo_starter_berhasil` dan dataset `Fashion-MNIST` agar asalnya jelas.
Publikasi hanya diizinkan jika artefak NIM belum ada atau masih placeholder/demo.
Metrics dan prediksi demo bukan hasil Transformer penghitung orang; penerimaan
bukti demo untuk tahap ini mengikuti ketentuan dosen.
