# Protokol evaluasi tim — Ruang Cerdas 5 Volt (Tim 31)

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request* baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Hitung orang di depan meja dari citra |
| Sumber dataset | Pengambilan citra secara mandiri oleh tim pada area meja pengujian; tautan dataset: belum tersedia |
| Jenis tugas (`TUGAS`) | Klasifikasi |
| Target / label | Jumlah orang di depan meja: 0, 1, 2, dan 3 |
| Ukuran data latih / validasi / uji | Belum ditentukan; menunggu dataset dan pembagian kelompok |
| Cara split | GroupKFold berdasarkan sesi, blok, atau partisipan sesuai ketentuan kartu kasus; implementasi perlu diselaraskan dengan starter |
| Metrik utama untuk komparasi | Accuracy dan MAE sesuai kartu kasus; perlu diselaraskan dengan evaluator starter yang menyediakan accuracy dan macro-F1 untuk klasifikasi |
| Praproses yang sama untuk semua model | Citra berukuran 224 × 224 piksel; membuang satu detik pertama setiap kondisi; normalisasi menggunakan statistik data latih saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop (atau yang disetujui pembimbing) |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | Belum tersedia; diisi setelah dataset dan manifest siap serta baseline berhasil dijalankan |

Disetujui pembimbing pada: belum disetujui (tautan *pull request* belum tersedia).