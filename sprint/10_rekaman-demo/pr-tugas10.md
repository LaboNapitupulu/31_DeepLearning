# Demo Sprint 1 Tim 31 dan reproduksi Windows (#10)

Video demo: [DemoS1_Kel31.mp4 di Google Drive](https://drive.google.com/file/d/1SSENODW6TGfoZSxkIcr_du-m5E-u_YLp/view?usp=sharing).

Dokumentasikan video tugas #10 beserta perintah reproduksi di README utama
dan folder `sprint/10_rekaman-demo/`. Runner kini meneruskan UTF-8 ke proses
anak agar evaluator dapat mencetak log di Windows. Sertakan hasil primer
terbaru setelah runner dijalankan pada dataset yang sama, serta bukti uji
integrasi terpisah tanpa menimpa arsip run sebelumnya.

Validasi: keenam komponen memakai seed 1, hash
`ae9811140ba30965a9d7c5e94fabd85d`, dan 40 citra validasi. Metrik dihitung
ulang dari prediksi tersimpan. Data uji G4 tidak dievaluasi. Data citra,
bobot pralatih, dan checkpoint model tidak disertakan di Git.

Perubahan prediksi CNN terhadap artefak awal tim dicatat: akurasi tetap
47,5%, sedangkan macro-F1 lokal 0,379570 dan MAE 0,725. TL tetap memiliki
akurasi 65%, macro-F1 0,431818, dan MAE 0,425 orang. Satu seed dan dataset
kurasi belum menjadi komparasi akhir Sprint 2; identitas partisipan 18
foto pelengkap masih memerlukan konfirmasi tim.

Tautan video diberikan oleh Labo. Status tugas dan unggahan tautan di
Papan Tim diperbarui terpisah.
