# 05 · Transfer Learning

Kartu Workbench #13–#14 · Labo Napitupulu (123450037).

[Model MobileNetV2](../../models/123450037/run.py) memakai bobot awal ImageNet pada run primer. Pada 40 citra validasi G1 dengan seed 1, akurasi mencapai **65,00%**, macro-F1 **0,4318**, dan MAE **0,4250 orang**.

- [Catatan model](../../models/123450037/catatan.md) dan [laporan individu](../../reports/individu/123450037.md)
- [Metrik](../../results/123450037/metrics.json), [detail](../../results/123450037/detail.json), dan [prediksi](../../results/123450037/pred_val.npy)

## Tugas #13: uji jalur TL pada simulasi

Dari akar repositori, memakai lingkungan Python yang berisi dependensi proyek:

```powershell
.\.venv\Scripts\python.exe models/123450037/run.py --seed 1 --simulasi --bobot data/mobilenet_v2-7ebf99e0.pth
```

Bobot ImageNet resmi bernama `mobilenet_v2-7ebf99e0.pth` disimpan di `data/`.
Jika berkas lokal belum tersedia, hilangkan `--bobot` agar torchvision mengunduh
bobot pralatih. Helper menyimpan cache unduhan di `.venv/torch-cache/`.

Perintah menjalankan [helper simulasi](../../models/123450037/uji_simulasi.py),
yang membuat salinan kode dan manifest dalam direktori baru di
`results/simulasi/123450037/seed_1-*/`. Data sintetis dibangkitkan oleh modul tim,
lalu `models/123450037/run.py --seed 1` dijalankan pada salinan itu.
Tidak bergantung pada opsi simulasi di `run_all.py`.

Di dalam direktori hasil tersebut tersedia:

- `data/okupansi/`: dataset sintetis dan provenance, 160 latih / 80 validasi / 80 uji.
- `results/123450037/`: prediksi validasi, metrics dari evaluator, detail, dan checkpoint.
- `run.log` dan `ringkasan.json`: bukti eksekusi, versi dependensi, dan pemeriksaan.

Masukan model RGB uint8 224x224, keluaran empat kelas 0-3. Data uji tidak
dievaluasi. Helper memeriksa bahwa berkas data dan hasil primer tetap memiliki
SHA-256 yang sama setelah simulasi selesai.

Angka sintetis hanya membuktikan jalur berjalan. Hasil primer pada bagian atas
berasal dari artefak tim dan dilaporkan terpisah dari uji lokal tugas #13.
Data, bobot, dan checkpoint diabaikan Git; daftar berkas PR mencakup kode,
dokumentasi, dan ringkasan uji.

## Bukti uji lokal 27 November 2026

Seed 1 dengan bobot MobileNetV2 ImageNet, lima epoch, backbone beku,
batch 16, AdamW LR 0,001, weight decay 0,01, dropout 0,2, dan empat thread CPU:

| Data | Validasi | Akurasi | Macro-F1 | MAE orang | Durasi run |
|---|---:|---:|---:|---:|---:|
| Sintetis tim | 80 citra | 97,50% | 0,968627 | 0,025 | 19,694 detik |

[Ringkasan yang dapat di-commit](hasil-simulasi.json) mencatat versi dependensi,
hash bobot, dan folder hasil lengkap. Run sukses disimpan di
`results/simulasi/123450037/seed_1-93pbs740/`. Pemeriksaan ulang metrik,
bentuk masukan/keluaran, pemuatan checkpoint, dan penjagaan artefak primer
dilakukan. Tidak ada evaluasi data uji.

Run pertama berhenti saat terminal Windows tidak dapat menampilkan simbol
Unicode dari evaluator. Helper kini memakai pengganti karakter yang tidak
didukung untuk tampilan terminal, sedangkan log berkas tetap UTF-8. Run
diulang sampai seluruh pipeline dan pemeriksaan selesai.

## Tugas #14: run lokal pada data primer

Pada 27 November 2026, konfigurasi awal dilatih ulang di laptop Labo pada
dataset primer dengan hash `ae9811140ba30965a9d7c5e94fabd85d`.
Latih G2/G3 berisi 80 citra, validasi G1 40 citra, dan G4 40 citra disimpan
untuk evaluasi akhir. Data uji tidak dievaluasi pada run ini.

```powershell
.\.venv\Scripts\python.exe -u -B models/123450037/run.py --seed 1 --epochs 5 --batch-size 16 --lr 0.001 --weight-decay 0.01 --dropout 0.2 --threads 4 --perangkat "CPU laptop Labo - Windows" --bobot data/mobilenet_v2-7ebf99e0.pth
```

| Model | Seed | Akurasi validasi | Macro-F1 | MAE orang | Durasi tercatat |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 ImageNet, backbone beku | 1 | 65,00% | 0,431818 | 0,425 | 11,758 detik |
| Histogram RGB + HOG + Random Forest | 1 | 60,00% | 0,541259 | 0,600 | 0,515 detik |

Pembanding memakai hash dan split validasi yang sama. Prediksi TL lokal
identik dengan artefak awal yang diterima dari tim. Hasil primer utama dan
`seed_1/` kini mencatat run lokal; artefak tim sebelumnya disimpan di
`results/123450037/arsip_tim_seed_1/`.

- [Ringkasan verifikasi primer](hasil-primer-lokal.json) dan [log eksekusi](run-primer-lokal.log).
- [Draf Log Sprint 1](log-sprint-1.md) untuk dimasukkan ke Workbench.
- [Draf judul/deskripsi PR #14](pr-tugas14.md).

Semua 26 prediksi yang benar berasal dari kelas 0 dan 1. Recall validasi
kelas 2 dan 3 masih nol, sehingga akurasi 65% belum menunjukkan kemampuan
menghitung seluruh kelas. Angka ini satu seed; tiga seed, tuning, ablation,
ONNX, dan latensi menjadi pekerjaan Sprint 2. Identitas partisipan pada 18
foto pelengkap tetap perlu dikonfirmasi sesuai catatan mutu data tim.

Pemeriksaan skema dan isi metrics dilakukan secara lokal. Pemeriksaan
artefak melalui Workbench, pengisian Log Sprint, serta pengajuan PR belum
dilakukan oleh skrip ini.
