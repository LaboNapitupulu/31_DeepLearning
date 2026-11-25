# Catatan mutu data Tim 31

## Penyesuaian dataset

Tim menyesuaikan dataset menjadi **tepat 160 citra JPEG**, dengan 40 citra
per kelompok dan pembagian 80 latih, 40 validasi, serta 40 uji.
Folder kerja kamera dan kedua indeks memuat 160 entri yang sama.
Nama JPEG mengikuti nomor slot pada log asli. Waktu pengambilan tetap berasal
dari rekaman kamera. Model memakai kolom `okupansi` pada `anotasi_visual.csv`.
Terdapat 20 foto dengan jumlah orang berbeda dari slot rancangan. Bila posisi
duduk atau berdiri dan status lampu ikut dihitung, terdapat 35 selisih.

| Kelompok | 0 orang | 1 orang | 2 orang | 3 orang | Split |
|---|---:|---:|---:|---:|---|
| G1 | 10 | 16 | 9 | 5 | Validasi |
| G2 | 14 | 12 | 7 | 7 | Latih |
| G3 | 12 | 15 | 10 | 3 | Latih |
| G4 | 11 | 12 | 10 | 7 | Uji |

Seleksi meminimalkan jumlah kuadrat selisih dari target per kelas sambil
menjaga 40 foto di tiap kelompok. Toleransi ±10% membutuhkan 18–22 foto per
kelas latih dan 9–11 per kelas validasi serta uji. Untuk mencapai toleransi
dengan stok saat ini diperlukan minimal 8 foto tambahan kelas 3 dan 1 kelas
2 pada latih, 4 foto kelas 3 pada validasi, serta 2 foto kelas 3 pada uji.

Ada 18 foto pelengkap dari luar blok terdekat, dengan 5 untuk G2, 5 untuk
G3, 4 untuk G1, dan 4 untuk G4. Identitas partisipan pada foto pelengkap
belum dikonfirmasi dari lembar sesi. Indeks mentah dan JPEG asli tersedia
dalam TAR kamera.

## Berkas dataset Sprint 1

| Berkas | Isi | Jumlah |
|---|---|---:|
| `kamera/` | Foto JPEG yang digunakan pada Sprint 1 | 160 foto |
| `kamera_index.csv` | Nama JPEG dan waktu pengambilan | 160 entri |
| `anotasi_visual.csv` | Label okupansi berdasarkan isi foto | 160 entri |
| `label.csv` dari `tim31_sesi-20261123-120519.tar.gz` | Log kondisi rancangan dengan seed 1031 | 160 kondisi |

## Arsip kamera mentah

`tim31_web-20261123-120524-ed41.tar` berisi 193 foto sebelum penyesuaian, dengan
interval pengambilan median 4,9842 detik. Kamera memotret pada interval tetap,
sehingga jumlah foto mentah berbeda dari jumlah kondisi log. Tim menyesuaikan
dataset Sprint 1 menjadi tepat 160 foto. Sebanyak 33 foto lain tersimpan
dalam arsip sumber.

## Arsip lain

Sesi `tim31_sesi-20261123-112233.tar.gz` adalah eksperimen Tim 31 yang terhenti
setelah 28 kondisi G3. Jendela kondisinya 11.22.57–11.25.03 WIB. Pasangan citra
yang dipakai Sprint 1 mulai 12.05.29 WIB. Rekaman 19 November di `data/kamera/`
adalah sesi terpisah.
