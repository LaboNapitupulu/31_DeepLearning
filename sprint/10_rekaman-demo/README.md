# Kartu #10 · Rekaman Demo Sprint 1

Penanggung jawab: Labo Napitupulu (123450037), Tim 31.

## Video demo untuk pengumpulan

[Tonton Demo Sprint 1 Tim 31 di Google Drive](https://drive.google.com/file/d/1SSENODW6TGfoZSxkIcr_du-m5E-u_YLp/view?usp=sharing).

Tautan diberikan oleh Labo untuk tugas #10. Metadata Google Drive terbaca
sebagai `DemoS1_Kel31.mp4`. Pengunggahan tautan ke Papan Tim dilakukan
terpisah; dokumen ini tidak menyatakan status tugas di platform berubah.

Video lokal lama `demo-sprint-1.mp4` adalah draf dari paket tim; tautan
Drive di atas menjadi rujukan video untuk pengumpulan.

## Bukti dan reproduksi

- [Log primer awal dari paket tim](run-primer.log)
- [Hasil komparasi](../../reports/hasil_komparasi.md)
- [Panduan merekam dan naskah](panduan-rekaman.md)
- [Bukti uji runner lengkap lokal](hasil-integrasi-lokal.json) dan [log baru](run-integrasi-lokal.log)
- [Tabel hasil run integrasi lokal](hasil-komparasi-lokal.md)
- [Perintah reproduksi](../../README.md#menjalankan-ulang-di-windows)

Uji integrasi awal dilakukan sebelum tautan video diberikan, pada salinan
kode dan data terpisah. Setelah
itu runner juga dijalankan di folder proyek. Tabel utama sekarang mengikuti
artefak primer terbaru di `results/`, termasuk durasi run tersebut.
Ringkasan dan log uji terpisah tetap dipertahankan sebagai bukti sebelumnya.

Keenam komponen berhasil dievaluasi pada validasi. CNN lokal memiliki satu
prediksi berbeda dari artefak awal tim: akurasi tetap 47,5%, macro-F1
0,379570 dan MAE 0,725. Perbedaan dicatat apa adanya. Evaluasi model pada
data uji G4 tidak dijalankan untuk demo Sprint 1.
