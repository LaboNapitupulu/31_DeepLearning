# Protokol evaluasi Tim 31

Status dokumen adalah draf untuk diajukan ke pembimbing lewat pull request. Data primer
yang diajukan harus diperiksa dan catatan sesi diperbarui sebelum protokol dibekukan.

| Butir | Ketentuan |
|---|---|
| Sumber | Arsip eksperimen dan indeks kamera Tim 31 dari Meja 5 Volt Workbench. |
| Target | Dataset yang telah disesuaikan memakai kelas 0–3 dari kolom `okupansi` pada anotasi visual. Duduk dan berdiri untuk satu orang sama-sama kelas 1. |
| Pencocokan | Setiap JPEG memiliki satu anotasi. Waktu pengambilannya diperiksa terhadap indeks kamera. Pemrosesan rekaman mentah memakai jendela `t_mulai_host + 1 ≤ t_host < t_selesai_host` dan label rancangan. |
| Praproses | Koreksi orientasi EXIF, konversi RGB, ubah ukuran menjadi 224×224 dengan interpolasi bilinear. Pemuat mengembalikan array uint8 NHWC. |
| Unit pembagian | G1–G4. Peserta dalam satu kelompok harus unik antargrup. Semua citra suatu kelompok berada pada satu bagian. |
| Split beku | Gunakan `src.common.split.indeks` pada **daftar kelompok**, seed 2026. G2/G3 untuk latih, G1 untuk validasi, G4 untuk uji. |
| Normalisasi | Model jaringan menghitung rerata dan simpangan baku kanal dari data latih. |
| Metrik | Akurasi dan MAE dalam satuan orang. Macro-F1 dicatat oleh evaluator starter. |
| Sprint 1 | Validasi dengan seed 1 pada data primer untuk setiap jalur dan pembanding. |
| Sprint 2 | Tiga seed pada konfigurasi akhir. Catat tuning, minimal dua variasi, serta ukuran ONNX dan latensi laptop meja. |
| Data uji | Dievaluasi sekali pada sprint terakhir oleh Penjaga Data. |
| Anggaran | Maksimal 20 menit per pelatihan di CPU laptop, mengikuti arahan pembimbing. |
| Sidik data | SHA-256 starter mencakup NPZ dan provenance. NPZ menyimpan citra dan indeks split. Provenance mencatat identitas sesi dan SHA-256 sumber setiap citra, indeks dan label. |

Dataset disesuaikan menjadi 160 citra dengan pembagian 80 latih, 40 validasi,
dan 40 uji. Delapan belas foto
pelengkap di luar blok memakai kelompok alokasi; identitas partisipannya
belum dikonfirmasi dari lembar sesi. Hasil validasi menggunakan pembagian
kelompok tersebut. Distribusi kelas dan selisih dari target tercatat di
`data/CATATAN-SESI.md`. Run awal Sprint 1 memakai 96 citra dengan `data_hash`
`689ba8e82130836bd05fccc993943d06`. Sumber dan pemeriksaan mutu dicatat di
`data/CATATAN-SESI.md`. Pada jalur rekaman mentah, kondisi yang terganggu
dapat dikeluarkan lewat `--keluarkan` dengan nomor pasangan, nomor kondisi,
dan alasan.
