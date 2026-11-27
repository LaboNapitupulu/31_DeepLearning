# Transfer learning MobileNetV2 · 123450037

Ekstraktor MobileNetV2 ImageNet dibekukan pada konfigurasi awal dan classifier
empat kelas dilatih dengan AdamW (LR 0,001), dropout 0,2, dan weight decay 0,01.
Dengan pembekuan, jumlah parameter yang diperbarui lebih kecil sehingga run awal
lebih ringan. Opsi `--fine-tune` membuka ekstraktor untuk percobaan berikutnya.
Rerata dan simpangan baku kanal dihitung dari citra latih sesuai protokol tim.

Untuk tugas #13, jalankan `python models/123450037/run.py --seed 1 --simulasi
--bobot data/mobilenet_v2-7ebf99e0.pth`. Helper menyalin kode ke direktori baru di
`results/simulasi/123450037/`, membangkitkan data sintetis tim, lalu melatih dan
menilai MobileNetV2 memakai pemuat dan evaluator yang sama. Statistik normalisasi
dihitung dari split latih simulasi. Data primer dan artefak hasil primer tidak
ditimpa; sidik berkasnya diperiksa sebelum dan sesudah eksekusi.

Bobot ImageNet digunakan pada uji lokal. `--bobot-acak-simulasi` hanya tersedia
untuk uji mekanisme tanpa pralatih dan tidak boleh dipakai pada data primer.
Simulasi tidak menerima `--final`. Bukti tugas #13, perintah reproduksi, dan
batas interpretasi hasil ada di `sprint/05_transfer-learning/README.md`.

Kandidat Sprint 2 adalah menguji pembekuan ekstraktor dan dropout. Run primer
seed 1 yang diterima dari tim mencatat akurasi 65,00% dan MAE 0,4250 orang di
validasi; uji simulasi lokal tidak mengganti hasil tersebut.

## Run primer lokal tugas #14

Pada 27 November 2026, seed 1 dilatih ulang dengan konfigurasi yang sama
pada data primer (80 latih G2/G3, 40 validasi G1). Prediksi validasi identik
dengan hasil tim: akurasi 65%, macro-F1 0,431818, MAE 0,425 orang. Durasi
tercatat di laptop Labo 11,758 detik. Data uji G4 tidak dievaluasi.

Pembanding Histogram RGB + HOG + Random Forest pada hash yang sama mencapai
60%, macro-F1 0,541259, MAE 0,600. TL unggul 5 poin persentase akurasi dan
MAE lebih rendah 0,175 orang, tetapi macro-F1 lebih rendah. Seluruh contoh
kelas 2 dan 3 pada validasi masih salah. Jangan memakai angka simulasi
97,5% untuk menyatakan kemampuan pada data primer.

Artefak primer utama sekarang berasal dari run lokal. Artefak awal tim
diarsipkan di `results/123450037/arsip_tim_seed_1/`. Log dan pemeriksaan
ulang metrics serta checkpoint ada di `sprint/05_transfer-learning/`.
