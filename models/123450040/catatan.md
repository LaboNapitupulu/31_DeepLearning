# CNN dari nol · 123450040

Tiga blok konvolusi mempelajari ciri lokal, mulai dari tepi hingga pola yang lebih
luas. Pooling spasial 2×2 sebelum classifier mempertahankan informasi kasar tentang
letak pola di dalam citra. Konfigurasi awal memakai AdamW (LR 0,001), weight decay
0,01, dan dropout 0,2. AdamW mengadaptasi langkah pembaruan per parameter dan memberi
kontrol weight decay yang terpisah.

Jalur diuji dengan `python run_all.py --simulasi --hanya 123450040 --tanpa-baseline`.
Untuk Sprint 2, kandidat variasi adalah penghilangan dropout dan pooling 1×1, dengan
split serta seed yang sama. Run primer seed 1 mencatat akurasi 47,50% dan MAE 0,7000 orang di validasi.
