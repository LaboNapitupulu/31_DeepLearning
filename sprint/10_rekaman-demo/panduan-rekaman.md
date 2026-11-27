# Naskah rekaman Labo · Demo Sprint 1 tugas #10

Baca hanya bagian **UCAPKAN**. Bagian **BUKA**, **SOROT**, dan **LANJUT**
adalah petunjuk tindakan, bukan kalimat untuk dibaca.
Ikuti langkah 1 sampai 8 secara berurutan. Waktu bicara sekitar empat menit;
waktu eksekusi runner dapat menambah durasi video.

## Sebelum menekan Record

Buka tab-tab ini dalam urutan berikut agar tinggal berpindah:

1. `README.md`.
2. Terminal PowerShell pada akar proyek.
3. `src/protokol/siapkan.py`, fungsi `muat_citra`.
4. `src/protokol/data.py`, fungsi `muat`.
5. `src/latih_citra.py`, baris `mean, std = statistik_latih(xtr, args.batch_size)`.
6. `models/123450037/run.py`, kelas `MobileNetOkupansi`.
7. Pratinjau `reports/hasil_komparasi.md`.

Di OBS pilih sumber Display Capture atau Window Capture yang menampilkan
editor beserta terminalnya. Periksa mikrofon dan lakukan tes singkat.
Perbesar teks agar terbaca. Rekam ke MKV, kemudian gunakan File > Remux
Recordings untuk menghasilkan MP4. Simpan video baru dengan nama
`demo-sprint-1-labo.mp4`.
Rujukan: [OBS Quick Start](https://obsproject.com/kb/quick-start-guide) dan
[Recording Output](https://obsproject.com/kb/standard-recording-output-guide).

## 1 · Perkenalan (sekitar 20 detik)

**BUKA:** README proyek, bagian judul dan paragraf pertama.

**UCAPKAN:**

> Saya Labo Napitupulu, NIM 123450037, dari Tim 31. Proyek kami menghitung
> jumlah orang di area meja ke dalam kelas nol sampai tiga melalui citra
> kamera. Saya akan menunjukkan data primer, pemrosesan data, pelatihan
> seluruh jalur, dan hasil perbandingannya pada Sprint 1.

**LANJUT:** pindah ke terminal.

## 2 · Buktikan data dan pembagiannya (sekitar 30 detik)

**BUKA:** terminal. Jalankan:

```powershell
.\.venv\Scripts\python.exe -B sprint/10_rekaman-demo/periksa_data.py
```

**SOROT:** keluaran `Jenis data`, `Data hash`, dan baris `train`, `val`, `test`.

**UCAPKAN:**

> Dataset primer yang dipakai berisi 160 citra kurasi. G2 dan G3 menjadi
> 80 citra latih. G1 menjadi 40 citra validasi, sedangkan G4 menjadi 40
> citra uji. Semua model memakai data dengan hash yang sama. Bagian uji
> disimpan untuk evaluasi akhir.

**LANJUT:** setelah keluaran selesai terlihat, buka `siapkan.py`.

## 3 · Jelaskan perubahan ukuran citra (sekitar 15 detik)

**BUKA:** `src/protokol/siapkan.py`. Gunakan Ctrl+F untuk mencari `def muat_citra`.

**SOROT:** dua baris yang mengandung `convert("RGB")` dan `rgb.resize(...)`.

```python
rgb = ImageOps.exif_transpose(gambar).convert("RGB")
citra = np.asarray(rgb.resize(BENTUK[:2], Image.Resampling.BILINEAR))
```

**UCAPKAN:**

> Pada tahap penyiapan, orientasi citra diperbaiki, citra dikonversi
> menjadi RGB, lalu ukurannya diubah menjadi 224 kali 224 piksel.
> Hasilnya disimpan sebagai dataset siap pakai.

**LANJUT:** pindah ke `data.py`. Penjelasan resize selesai di sini.

## 4 · Jelaskan pemuat data (sekitar 25 detik)

**BUKA:** `src/protokol/data.py`, fungsi `muat()`.

**SOROT:** mulai `with np.load(...)` sampai akhir `return tuple(...)`.

```python
with np.load(path, allow_pickle=False) as data:
    X, y = data["X"], data["y"]
    return tuple(
        array[data[bagian]]
        for bagian in ("train", "val", "test")
        for array in (X, y)
    )
```

**UCAPKAN:**

> Fungsi muat membaca citra X dan label y dari dataset yang sudah
> disiapkan. Indeks latih, validasi, dan uji sudah tersimpan di dalam
> dataset. Fungsi ini mengembalikan enam array: citra dan label untuk
> masing-masing bagian. Semua model memakai pemuat yang sama.

**LANJUT:** pindah ke `latih_citra.py`. Fungsi ini membaca split yang sudah
ada; tidak membuat split acak baru, mengubah ukuran, atau menghitung normalisasi.

## 5 · Jelaskan normalisasi (sekitar 15 detik)

**BUKA:** `src/latih_citra.py`. Cari dan sorot:

```python
mean, std = statistik_latih(xtr, args.batch_size)
```

**UCAPKAN:**

> Rerata dan simpangan baku kanal dihitung hanya dari citra latih,
> yaitu xtr. Statistik yang sama digunakan untuk normalisasi masukan
> model pada validasi, sehingga statistik data validasi tidak digunakan
> untuk menghitung normalisasi.

**LANJUT:** pindah ke kode MobileNetV2.

## 6 · Jelaskan model Anda (sekitar 25 detik)

**BUKA:** `models/123450037/run.py`, kelas `MobileNetOkupansi`.

**SOROT:** `self.fitur.requires_grad_(not bekukan)` dan
`nn.Linear(mobilenet.last_channel, 4)`. Setelah itu, tunjukkan bagian
`buat_model` yang memuat bobot.

**UCAPKAN:**

> Jalur saya memakai MobileNetV2 pralatih ImageNet. Pada konfigurasi
> awal, ekstraktor fiturnya dibekukan dan classifier dilatih untuk
> menghasilkan empat kelas. Konfigurasinya menggunakan AdamW dengan
> dropout dan weight decay sebagai regularisasi.

**LANJUT:** kembali ke terminal untuk menjalankan seluruh jalur.

## 7 · Jalankan seluruh jalur (durasi mengikuti proses)

**BUKA:** terminal. Jalankan:

```powershell
.\.venv\Scripts\python.exe -u -B run_all.py --seeds 1 --bobot-tl data/mobilenet_v2-7ebf99e0.pth
```

**UCAPKAN SAAT PROSES BERJALAN:**

> Perintah ini menjalankan baseline starter, Random Forest dengan HOG
> dan histogram warna, MobileNetV2, CNN, Transformer, serta GRU.
> Semua jalur menggunakan seed satu, pemuat bersama, dan evaluator yang
> sama. Evaluasi pada run ini memakai data validasi.

**SOROT:** sebagian loss per epoch, keluaran metrik setiap jalur, dan
pesan `Tabel komparasi` di akhir.

**LANJUT:** tunggu sampai pesan `Tabel komparasi` muncul dan terminal
kembali menerima perintah. Baru buka ulang pratinjau hasil komparasi.
Waktu menunggu boleh dipercepat saat mengedit dengan keterangan
"rekaman dipercepat". Perintah tidak memakai `--final`.

## 8 · Jelaskan hasil dan tutup (sekitar 45 detik)

**BUKA:** pratinjau `reports/hasil_komparasi.md` yang baru dihasilkan.

**SOROT:** baris `tl`, baris `rf_hog_color`, kolom akurasi, macro-F1,
MAE, dan kolom data uji yang masih berupa tanda kosong.

**UCAPKAN:**

> Pada validasi, MobileNetV2 mencapai akurasi 65 persen dan MAE 0,425
> orang. Random Forest mencapai akurasi 60 persen dan MAE 0,600.
> Namun, macro-F1 Random Forest lebih tinggi, sehingga keunggulan TL
> belum merata pada seluruh kelas.
>
> Pada validasi TL saat ini, kelas dua dan tiga masih salah seluruhnya.
> Hasil ini baru satu seed. Sprint 2 akan melanjutkan tuning, ablation,
> tiga seed, serta pengukuran ONNX dan latensi. Tim juga masih perlu
> mengonfirmasi keanggotaan partisipan pada foto pelengkap. Data uji G4
> belum dievaluasi. Terima kasih.

**LANJUT:** berhenti berbicara, tunggu sekitar dua detik, lalu Stop Recording.
Jika angka run baru berbeda, bacakan angka yang benar-benar muncul;
jangan mengubah tabel untuk mencocokkan naskah. Durasi pada kolom terakhir
mengikuti run terbaru dan dapat berubah.

## Sesudah merekam

Putar video untuk memeriksa suara dan keterbacaan. Pastikan terlihat
perintah yang dijalankan, hasil setiap jalur, dan tabel yang dihasilkan.
Unggah video atau tautannya ke Papan Tim. Perintah reproduksi sudah
tersedia di README utama.

## Catatan untuk persiapan, bukan dibaca seluruhnya

Dataset primer ini memakai anotasi visual pada kolom `okupansi`.
`siapkan.py` juga mendukung log mentah dengan jendela mulai satu detik
setelah kondisi; jalur itu berbeda dari kurasi yang sedang dipakai.
Catatan tim menyebut 18 foto pelengkap dari luar blok, sehingga konfirmasi
identitas partisipannya masih diperlukan. Klaim pemisahan yang diperiksa
oleh skrip adalah pemisahan ID kelompok dalam metadata.

Video lama `demo-sprint-1.mp4` berasal dari paket tim dan tetap disimpan
sebagai draf. Tautan video yang diberikan Labo untuk pengumpulan tersedia
di README folder tugas #10.
