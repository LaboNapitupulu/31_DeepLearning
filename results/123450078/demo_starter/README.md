# Bukti demo starter dosen

Dataset: Fashion-MNIST lokal dengan contoh pemuat bawaan starter.
Model: baseline regresi logistik NumPy dari starter.
Hasil ini untuk membuktikan alur starter berjalan; kasus tim tetap penghitung orang.

- Seed: 2026
- Ukuran validasi: 6000
- Akurasi validasi: 0.842333
- Macro-F1 validasi: 0.841142
- Durasi baseline (detik): 1.4
- Data hash: `42d7ac7f6bd298d04c19a58de0bb6a97`
- Data uji tidak dievaluasi.

`metrics.json` disalin tanpa perubahan dari evaluator starter; `pred_val.npy`
berisi prediksi asli. Detail reproduksi ada di `demo.json` dan log di `run.log`.

```bash
python3 models/123450078/demo_starter.py --seed 2026 --threads 4 --publish-demo
```

Penerimaan demo untuk tahap pengumpulan mengikuti ketentuan dosen.
