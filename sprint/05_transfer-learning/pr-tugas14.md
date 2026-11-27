# Judul

Reproduksi TL seed 1 pada data primer untuk tugas #14

# Deskripsi

Jalur MobileNetV2 ImageNet kini memiliki bukti pelatihan lokal pada data
primer Tim 31. Entry point TL mengatur UTF-8 agar proses evaluator tidak
gagal ketika mencetak simbol Unicode pada terminal Windows.

Seed 1, lima epoch, backbone beku, dan split G2/G3 untuk latih serta G1
untuk validasi menghasilkan akurasi 65%, macro-F1 0,431818, dan MAE 0,425
orang. Prediksi identik dengan hasil tim sebelumnya. Data uji G4 tidak
dievaluasi. Artefak tim sebelumnya diarsipkan; artefak utama mencatat
run lokal, sementara checkpoint dan data primer tetap diabaikan Git.

Pembanding non-DL pada hash yang sama mencapai 60% dan MAE 0,600.
Recall TL kelas 2 dan 3 masih nol; hasil satu seed ini belum menjadi
perbandingan akhir Sprint 2.

Validasi: hash dataset dan bobot, metrik dari prediksi validasi, salinan
hasil seed, pemuatan checkpoint dan logits empat kelas, serta tidak
adanya evaluasi uji diperiksa. Log dan ringkasan tersedia di
`sprint/05_transfer-learning/`.

Hash data: `ae9811140ba30965a9d7c5e94fabd85d`.

Keanggotaan partisipan pada 18 foto pelengkap masih memerlukan konfirmasi
tim sesuai `data/CATATAN-SESI.md`. Pemeriksaan artefak dan Log Sprint di
Workbench dilakukan terpisah dari validasi lokal ini.
