# RNN atas baris citra · 123450091

Average pooling 4×4 menghasilkan citra 56×56. Setiap baris diproyeksikan ke vektor
berukuran 64 dan dibaca GRU sebagai satu langkah urutan spasial. Keadaan tersembunyi
terakhir diproyeksikan ke empat kelas. Konfigurasi awal memakai AdamW (LR 0,001),
dropout 0,2, dan weight decay 0,01 untuk mengendalikan pembelajaran pada data kecil.

Jalur diuji dengan `python run_all.py --simulasi --hanya 123450091 --tanpa-baseline`.
Kandidat Sprint 2 adalah mengubah ukuran pooling dan mengevaluasi dampak dropout.
Run primer seed 1 mencatat akurasi 57,50% dan MAE 0,6250 orang di validasi.
