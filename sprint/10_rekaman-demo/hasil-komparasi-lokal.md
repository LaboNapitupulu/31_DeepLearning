# Komparasi okupansi · data primer

Sidik data: `ae9811140ba30965a9d7c5e94fabd85d`.
Jumlah citra latih/validasi/uji: 80 / 40 / 40.

| Komponen | Jalur | Seed | Val akurasi (%) | Val macro-F1 | Val MAE | Test akurasi (%) | Test MAE | Detik |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Starter | baseline | 1 | 25.00 | 0.1000 | 1.2250 | — | — | 0.40 |
| Pembanding | rf_hog_color | 1 | 60.00 | 0.5413 | 0.6000 | — | — | 1.36 |
| 123450037 | tl | 1 | 65.00 | 0.4318 | 0.4250 | — | — | 11.01 |
| 123450040 | cnn | 1 | 47.50 | 0.3796 | 0.7250 | — | — | 9.29 |
| 123450078 | trf | 1 | 42.50 | 0.3108 | 0.7250 | — | — | 8.66 |
| 123450091 | rnn | 1 | 57.50 | 0.4157 | 0.6250 | — | — | 6.72 |

Uji integrasi lokal Windows 27 November 2026. Hasil ini berasal dari salinan
kode dan data terpisah. Artefak primer utama tidak ditimpa. Versi paket dan
perbedaan terhadap artefak repo dicatat di `hasil-integrasi-lokal.json`.
