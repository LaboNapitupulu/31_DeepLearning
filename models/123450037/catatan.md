# Catatan model Labo Napitupulu (123450037) — Transfer Learning (tl)

## Alasan optimizer
Menggunakan **AdamW** dengan learning rate awal $1 \times 10^{-3}$ dan weight decay $1 \times 10^{-2}$.
Alasannya:
1. AdamW memisahkan regularisasi $L_2$ (weight decay) dari kalkulasi gradien adaptif, sehingga memperbarui bobot secara lebih stabil tanpa merusak fitur representasi umum yang sudah dipelajari oleh model *backbone* ImageNet.
2. Kecepatan konvergensinya lebih cepat dibandingkan SGD biasa pada dataset skala kecil-menengah, sangat krusial untuk memenuhi batas anggaran pelatihan $\le 20$ menit pada CPU laptop.

## Regularisasi
1. **Dropout (0.2):** Diterapkan pada classifier head sebelum lapisan proyeksi akhir untuk mematikan neuron secara acak dan mencegah model terlalu bergantung pada fitur spasial tertentu dari data meja uji.
2. **Weight Decay ($10^{-2}$):** Mencegah bobot lapisan klasifikasi tumbuh terlalu besar, mengurangi risiko *overfitting* terhadap variasi pencahayaan tertentu (kondisi 128 vs 255).

## Pencarian hyperparameter (tabel)
*(Akan dilengkapi saat data primer meja uji sudah direkam pada Sprint 1)*

| Eksperimen | Backbone | LR | Batch | Optimizer | Val Acc | Val MAE | Catatan |
|---|---|---|---|---|---|---|---|
| E0 (Baseline) | MobileNetV2 | 1e-3 | 16 | AdamW | — | — | Konfigurasi awal |
| E1 | MobileNetV2 | 5e-4 | 16 | AdamW | — | — | Uji stabilitas LR lebih kecil |
| E2 | ResNet18 | 1e-3 | 16 | AdamW | — | — | Pembanding backbone CNN |

## Ablation (>= 2)
1. **Ablation 1 (Frozen Backbone vs Full Fine-Tuning):** Membandingkan apakah membekukan (*freeze*) bobot ekstraktor fitur menghasilkan generalisasi yang lebih baik dibanding melatih seluruh lapisan secara langsung pada data meja uji.
2. **Ablation 2 (Dampak Regularisasi Dropout):** Menguji akurasi dan MAE model dengan Dropout 0.2 vs tanpa Dropout (0.0).

## Hasil 3 seed konfigurasi akhir
*(Akan dieksekusi dengan seed 1, 2, dan 3 pada konfigurasi terbaik)*

| Konfigurasi | Seed | Val Acc | Val MAE | Durasi (s) | Ukuran ONNX (KB) | Latensi (ms) |
|---|---|---|---|---|---|---|
| Final | 1 | — | — | — | — | — |
| Final | 2 | — | — | — | — | — |
| Final | 3 | — | — | — | — | — |
| **Rerata ± Std** | — | — | — | — | — | — |
