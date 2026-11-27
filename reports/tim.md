# Laporan Tim 31 · Sprint 1

## Ringkasan

Empat model klasifikasi okupansi dan pembanding Random Forest dengan HOG dan
histogram warna dilatih ulang pada 160 citra Meja 5 Volt yang telah dikurasi.
Pada 40 citra validasi, MobileNetV2 mendapat akurasi tertinggi sebesar 65,00%.
Random Forest mendapat macro-F1 tertinggi sebesar 0,5413.

## Protokol

Dataset 160 citra RGB 224×224 memakai label okupansi aktual dari
`anotasi_visual.csv`, dipasangkan melalui nama JPEG dan indeks kamera. G2 dan
G3 dipakai untuk latih (80), G1 untuk validasi (40), dan G4 disimpan untuk uji
(40). Seed pembagian kelompok 2026 dan seed pelatihan 1. Akurasi dan MAE
adalah metrik utama. Evaluator starter juga mencatat macro-F1.

## Hasil

| Model | Akurasi validasi | Macro-F1 | MAE (orang) | Prediksi benar |
|---|---:|---:|---:|---:|
| Random Forest + HOG + histogram | 60,00% | 0,5413 | 0,6000 | 24/40 |
| MobileNetV2 ImageNet | 65,00% | 0,4318 | 0,4250 | 26/40 |
| CNN dari nol | 47,50% | 0,3907 | 0,7000 | 19/40 |
| Transformer | 42,50% | 0,3108 | 0,7250 | 17/40 |
| GRU atas baris citra | 57,50% | 0,4157 | 0,6250 | 23/40 |

Prediksi tetap kelas mayoritas latih (kelas 1) menghasilkan akurasi 40,00%
dan MAE 0,7250 di validasi. Tabel eksekusi dan durasi tersedia di
`reports/hasil_komparasi.md`. Baseline starter memprediksi kelas 0 untuk
seluruh citra validasi dan menghasilkan akurasi 25,00%.

## Analisis

MobileNetV2 memprediksi benar semua citra kelas 0 dan 1. Random Forest
mencatat prediksi benar pada keempat kelas. Transformer mendapat 17 dari 40
prediksi benar dan melewatkan semua foto kelas 3. Dengan 40 citra validasi, satu
prediksi benar tambahan mengubah akurasi sebesar 2,5 poin persentase.

Dua puluh foto kurasi berbeda dari slot rancangan dalam jumlah orang.
Sebanyak 18 foto di luar blok memakai kelompok alokasi. Distribusi kelas
masih di luar toleransi ±10% karena stok foto kelas 2 dan 3 terbatas. Identitas partisipan
foto tersebut belum dikonfirmasi dari lembar sesi. Run lama memakai 96 citra
berlabel log, sehingga kedua run memakai sampel dan acuan evaluasi berbeda.

## Rekomendasi

Pemeriksaan kualitas label perlu diselesaikan sebelum memilih purwarupa.
Setelah itu, bandingkan tiga seed, minimal dua ablation per jalur, dan ukur
ukuran ONNX serta latensi pada laptop meja. Evaluasi G4 dilakukan setelah
konfigurasi dipilih.

## Pembagian kerja

| Anggota | NIM | Jalur | Peran |
|---|---|---|---|
| Labo Napitupulu | 123450037 | MobileNetV2 | Reproduksi dan Demo |
| Aprilia Dewi Hutapea | 123450040 | CNN | Analisis dan pembanding |
| Givaro Ananta | 123450078 | Transformer | Integrasi |
| Muhammad Ridwan | 123450091 | GRU atas baris citra | Data dan Evaluasi |
