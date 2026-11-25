# 01 · Rancangan dan slot

Berkas untuk ditinjau: `rencana/t31-okupansi-citra.yaml`, `blueprint.yaml`, dan
`sprint/01_rancangan/README.md`.

Rencana mencakup empat kelompok berbeda, lima konfigurasi orang, lampu 0 dan 255,
dan empat ulangan. Total 160 kondisi. Draf mengikuti kartu kasus dengan durasi
4 detik dan jeda 0,5 detik. Usulan jeda 3 detik masih menunggu persetujuan.

| Peran lapangan | Anggota sesuai catatan tim |
|---|---|
| Operator perekam dan alat | Givaro |
| Koordinator partisipan | Ridwan |
| Pencatat kejadian | Aprilia |
| Penjaga bingkai dan suara | Labo |

Lembar sesi berisi waktu, ID kondisi, kelompok, dan kejadian yang mengganggu.
Cocokkan 12 ID samaran dengan kelompok yang berbeda. Simpan daftar nama dan
formulir persetujuan bersama koordinator partisipan. Pengelola meja perlu menerima
YAML yang disetujui pembimbing.

Simulasi paket meja dari direktori perekam:

```bash
python meja.py --simulasi jalankan ../rencana/t31-okupansi-citra.yaml
```

Sesi sintetis proyek tersedia untuk menguji pemotong tanpa perangkat melalui
`python run_all.py --simulasi --seeds 1`.
