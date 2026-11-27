# Komparasi okupansi · data primer

Sidik data: `ae9811140ba30965a9d7c5e94fabd85d`.
Jumlah citra latih/validasi/uji: 80 / 40 / 40.

| Komponen | Jalur | Seed | Val akurasi (%) | Val macro-F1 | Val MAE | Test akurasi (%) | Test MAE | Detik |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Starter | baseline | 1 | 25.00 | 0.1000 | 1.2250 | — | — | 0.90 |
| Pembanding | rf_hog_color | 1 | 60.00 | 0.5413 | 0.6000 | — | — | 1.46 |
| 123450037 | tl | 1 | 65.00 | 0.4318 | 0.4250 | — | — | 12.22 |
| 123450040 | cnn | 1 | 47.50 | 0.3796 | 0.7250 | — | — | 8.47 |
| 123450078 | trf | 1 | 42.50 | 0.3108 | 0.7250 | — | — | 8.34 |
| 123450091 | rnn | 1 | 57.50 | 0.4157 | 0.6250 | — | — | 5.92 |
