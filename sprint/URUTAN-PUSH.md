# Urutan push Sprint 1 Tim 31

Satu kartu memakai satu folder, satu commit berkas kartu, dan satu PR ke main. Setiap anggota memakai branch kerja yang sudah ada. Tunggu PR sebelumnya masuk main sebelum lanjut kartu berikutnya pada branch yang sama.

| Anggota | Branch kerja |
|---|---|
| Givaro | `123450078/kerja` |
| Aprilia | `123450040/kerja` |
| Labo | `123450037/kerja` |
| Ridwan | `123450091/kerja` |

## Urutan yang masih perlu diajukan

| Urutan | Kartu | Anggota | Branch | Folder kartu |
|---|---|---|---|---|
| 1 | #15 · Transformer pada data simulasi | Givaro Ananta | `123450078/kerja` | `Givaro-Ananta/kartu-15_transformer-simulasi` |
| 1 · paralel | #17 · GRU pada data simulasi | Muhammad Ridwan | `123450091/kerja` | `Muhammad-Ridwan/kartu-17_gru-simulasi` |
| 2 | #16 · Transformer pada data primer | Givaro Ananta | `123450078/kerja` | `Givaro-Ananta/kartu-16_transformer-primer` |
| 2 · paralel | #18 · GRU pada data primer | Muhammad Ridwan | `123450091/kerja` | `Muhammad-Ridwan/kartu-18_gru-primer` |
| 3 | #6 · Runner tim pada data simulasi | Givaro Ananta | `123450078/kerja` | `Givaro-Ananta/kartu-06_runner-simulasi` |
| 4 | #7 · Integrasi data primer | Givaro Ananta | `123450078/kerja` | `Givaro-Ananta/kartu-07_integrasi-primer` |
| 5 | #9 · Gladi rencana | Labo Napitupulu | `123450037/kerja` | `Labo-Napitupulu/kartu-09_gladi` |
| 6 | #10 · Rekaman Demo Sprint 1 | Labo Napitupulu | `123450037/kerja` | `Labo-Napitupulu/kartu-10_rekaman-demo` |
| 7 | #3 · Sprint Goal dan Log Sprint | Givaro Ananta | `123450078/kerja` | `Givaro-Ananta/kartu-03_sprint-goal-log` |

## Kartu yang berkas utamanya sudah ada di main

| Kartu | Bukti yang tersedia |
|---|---|
| #1 | Rancangan perekaman |
| #2 | Pembagian peran slot |
| #4 | Kode pemotong dan protokol (PR #9) |
| #5 | Penyiapan data primer (PR #9) |
| #8 | Kode dan artefak pembanding Random Forest |
| #11 | Kode CNN |
| #12 | Metrik primer CNN seed 1 |
| #13 | Kode MobileNetV2 |
| #14 | Metrik primer MobileNetV2 seed 1 dari run Labo |

Status berkas diperiksa pada [main `6cef08f`](https://github.com/LaboNapitupulu/31_DeepLearning/commit/6cef08fd172078358f2842d3a0ca6cb38da71137). Kartu #8, #11, #12, #13, dan #14 tetap mempunyai folder terpisah sebagai arsip. Pertahankan versi MobileNetV2 yang sudah ada di main saat integrasi.

## Isi setiap kartu

| Kartu | Folder | Berkas utama di repo |
|---|---|---|
| #3 | `Givaro-Ananta/kartu-03_sprint-goal-log` | reports/tim.md dan sprint/09_demo-dan-log/ |
| #6 | `Givaro-Ananta/kartu-06_runner-simulasi` | run_all.py dan sprint/08_integrasi/simulasi.md |
| #7 | `Givaro-Ananta/kartu-07_integrasi-primer` | README repo, laporan komparasi dan hasil baseline starter |
| #8 | `Aprilia-Dewi-Hutapea/kartu-08_pembanding` | src/pembanding/ dan results/baseline/ |
| #9 | `Labo-Napitupulu/kartu-09_gladi` | sprint/09_gladi-rencana/ |
| #10 | `Labo-Napitupulu/kartu-10_rekaman-demo` | sprint/10_rekaman-demo/ |
| #11 | `Aprilia-Dewi-Hutapea/kartu-11_cnn-simulasi` | models/123450040/run.py dan sprint/04_cnn/simulasi.md |
| #12 | `Aprilia-Dewi-Hutapea/kartu-12_cnn-primer` | results/123450040/ dan laporan individu |
| #13 | `Labo-Napitupulu/kartu-13_transfer-learning-simulasi` | models/123450037/run.py dan sprint/05_transfer-learning/simulasi.md |
| #14 | `Labo-Napitupulu/kartu-14_transfer-learning-primer` | results/123450037/ dan laporan individu |
| #15 | `Givaro-Ananta/kartu-15_transformer-simulasi` | models/123450078/run.py dan sprint/07_transformer/simulasi.md |
| #16 | `Givaro-Ananta/kartu-16_transformer-primer` | results/123450078/ dan laporan individu |
| #17 | `Muhammad-Ridwan/kartu-17_gru-simulasi` | models/123450091/run.py dan sprint/06_rnn/simulasi.md |
| #18 | `Muhammad-Ridwan/kartu-18_gru-primer` | results/123450091/ dan laporan individu |

## Menyalin dan meninjau satu kartu

Masuk ke akar repo tim. Isi `paket_tim31` dengan path absolut folder hasil ekstrak `paket-anggota`. Contoh di bawah memilih kartu #15. Untuk kartu lain, ganti `tahap` dan `branch` sesuai tabel.

Folder `repo/` di dalam kartu mengikuti struktur tujuan di repositori. `berkas.txt` di sebelahnya adalah daftar untuk staging.

```bash
paket_tim31="/path/ke/paket-anggota"
tahap="$paket_tim31/Givaro-Ananta/kartu-15_transformer-simulasi"
branch="123450078/kerja"

test -z "$(git status --porcelain)" &&
  test -f "$tahap/berkas.txt" &&
  git fetch origin &&
  git switch "$branch" &&
  git merge --ff-only "origin/$branch" &&
  git merge --ff-only origin/main &&
  git diff --quiet origin/main HEAD &&
  cp -a "$tahap/repo/." . &&
  git add --pathspec-from-file="$tahap/berkas.txt" &&
  git diff --cached --check &&
  git diff --cached --name-status
```

Jalankan blok berikut setelah daftar staging sesuai kartu. Sesuaikan pesan commit dengan nomor kartu, lalu buka PR ke `main` dari branch anggota.

```bash
test "$(git branch --show-current)" = "$branch" &&
  test -f "$tahap/berkas.txt" &&
  test -z "$(LC_ALL=C comm -23 <(git diff --cached --name-only | LC_ALL=C sort) <(LC_ALL=C sort "$tahap/berkas.txt"))" &&
  git commit -m "Sprint 1: Transformer simulasi (#15)" &&
  git diff --name-status origin/main...HEAD &&
  git push origin "HEAD:refs/heads/$branch"
```

Jika fast-forward berhenti, lihat `git log --oneline --left-right origin/main...HEAD` dan cek status PR terakhir sebelum menyalin berkas. Branch kerja Ridwan masih punya satu commit lama di luar main saat diperiksa.

## Sinkronisasi pertama Ridwan

Jika isi commit lama Ridwan sudah boleh dikesampingkan, jalankan sinkronisasi ini dari working tree bersih sebelum kartu #17. Commit lama tetap tercatat di riwayat.

```bash
branch="123450091/kerja"
test -z "$(git status --porcelain)" &&
  git fetch origin &&
  git switch "$branch" &&
  git merge --ff-only "origin/$branch" &&
  test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$branch")" &&
  git merge -s ours --no-ff --no-commit origin/main &&
  git rev-parse --verify MERGE_HEAD &&
  git restore --source=origin/main --staged --worktree -- :/ &&
  git diff --cached --quiet origin/main &&
  git diff --quiet &&
  git commit -m "Sinkronkan branch kerja Ridwan dengan main" &&
  git diff --exit-code origin/main HEAD
```

## Reproduksi integrasi

Kartu #6 menyediakan runner simulasi. Kartu #7 menyimpan hasil run primer seed 1. Jalur Transfer Learning memakai berkas bobot MobileNetV2 ImageNet pada run primer.

```bash
python run_all.py --simulasi --seeds 1
python run_all.py --seeds 1 --bobot-tl data/mobilenet_v2-7ebf99e0.pth
```
