# Draf Log Sprint 1 · Labo Napitupulu (123450037)

Tanggal: 27 November 2026. Tugas #13 dan #14, jalur Transfer Learning.

Saya menjalankan MobileNetV2 ImageNet dengan backbone beku dan classifier
empat kelas. Tugas #13 telah diuji pada citra sintetis dengan seed 1; hasil
dan dataset sintetis disimpan terpisah dari data primer.

Untuk #14, saya melatih konfigurasi awal pada 80 citra primer G2/G3 selama
lima epoch, seed 1, batch 16, AdamW LR 0,001, weight decay 0,01, dropout
0,2, dan empat thread CPU. Hash dataset:
`ae9811140ba30965a9d7c5e94fabd85d`.

Pada 40 citra validasi G1, akurasi 65,00%, macro-F1 0,431818, dan MAE
0,425 orang. Durasi tercatat 11,758 detik. Prediksi lokal identik dengan
artefak awal tim. Data uji G4 tidak dievaluasi.

Pembanding Histogram RGB + HOG + Random Forest pada hash yang sama
mencapai akurasi 60%, macro-F1 0,541259, dan MAE 0,600. TL memiliki
akurasi lebih tinggi 5 poin persentase serta MAE lebih rendah 0,175,
tetapi macro-F1 lebih rendah. Recall kelas 2 dan 3 TL masih nol.

Saya memeriksa ulang metrik validasi, hash dataset, salinan artefak seed,
checkpoint, dan keluaran empat kelas. Penanganan UTF-8 ditambahkan pada
entry point TL agar evaluator dapat mencetak log di Windows.

Bukti ada di `sprint/05_transfer-learning/hasil-primer-lokal.json`,
`run-primer-lokal.log`, dan `results/123450037/`.

Tindak lanjut Sprint 2: tiga seed, fine-tuning dibanding backbone beku,
ablation dropout, serta ekspor dan pengukuran ONNX. Tim masih perlu
mengonfirmasi identitas partisipan pada 18 foto pelengkap.

Dokumen ini draf untuk disalin ke Workbench. Pemeriksaan artefak pada
platform, pengisian Log Sprint, dan pengajuan PR belum dinyatakan selesai.
