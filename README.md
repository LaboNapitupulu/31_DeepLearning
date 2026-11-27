# Tim 31 · Estimasi okupansi area meja

Proyek ini mengestimasi jumlah orang di area meja dari citra RGB 224×224 piksel. Targetnya empat kelas, 0 hingga 3 orang. Posisi duduk dan berdiri dikelompokkan dalam kelas 1.

## Berkas proyek

| Berkas | Isi |
|---|---|
| [blueprint.yaml](blueprint.yaml) | Rancangan pengambilan data, pembagian kelompok, dan jalur model |
| [Rancangan sesi](rencana/t31-okupansi-citra.yaml) | Konfigurasi 160 kondisi perekaman |
| [Catatan model](models/README.md) | Kode MobileNetV2 dan Transformer di checkout ini |
| [Laporan tim](reports/tim.md) | Berkas laporan tim |
| [Catatan sprint](sprint/README.md) | Rancangan dan lembar sesi |

## Hasil demo starter

[Demo Fashion-MNIST](results/123450078/demo_starter/README.md) mencatat akurasi validasi **84,23%** dan macro-F1 **0,8411** dari baseline regresi logistik pada 6.000 citra dengan seed 2026. Angka tersebut berasal dari demo Fashion-MNIST. Hasil data primer tim didokumentasikan di paket Sprint 1.

[Lisensi MIT](LICENSE)
