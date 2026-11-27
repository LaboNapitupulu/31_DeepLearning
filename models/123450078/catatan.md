# Transformer · 123450078

Citra RGB 224×224 dibagi menjadi 49 patch 32×32. Patch diproyeksikan ke dimensi 64,
ditambah embedding posisi, dan diproses oleh dua Transformer encoder, masing-masing
dengan empat head. Mean pooling, LayerNorm, dan classifier menghasilkan empat logits.
Statistik kanal berasal dari split latih dan disimpan bersama checkpoint.

Konfigurasi awal memakai AdamW dengan LR 0,0003, dropout 0,1, weight decay 0,01,
batch 16, dan 10 epoch. AdamW mengadaptasi langkah pembaruan sambil memisahkan
penalti weight decay. Perbandingan LR 0,0001 dan 0,0003 serta dua ablation
tanpa dropout dan tanpa weight decay disiapkan untuk Sprint 2.

```bash
python run_all.py --simulasi --hanya 123450078 --tanpa-baseline
python models/123450078/run.py --seed 1
```

Validasi memakai evaluator starter. MAE disimpan bersama detail setiap run.
Artefak Fashion-MNIST lama direkam sebagai demo starter pada Sprint 0. Run primer
seed 1 mencatat akurasi 42,50% dan MAE 0,7250 orang di validasi. Ukuran ONNX
dan latensi laptop meja diukur pada Sprint 2.
