# 03 · Pembanding non-DL

Berkas untuk ditinjau: `src/pembanding/` dan `sprint/03_pembanding/README.md`.

```bash
python -m src.pembanding.run --seed 1
```

Keluaran primer ada di `results/baseline/`. Masukan Random Forest adalah gabungan
HOG dan histogram RGB ternormalisasi pada citra 64×64. Evaluator starter menulis
akurasi dan macro-F1. MAE okupansi tersimpan di `detail.json`. Baseline starter dosen
dijalankan oleh integrator dan memiliki lokasi artefak `results/baseline_starter/`.
