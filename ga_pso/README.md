# Tugas Besar 1: Problem Solving Agent

**Mata kuliah:** AI5103 Artificial Intelligence and Data Science Fundamental

## Anggota Kelompok

| NIM | Nama |
| --- | --- |
| 28426006 | Muhammad Faishal Dienul Haq |
| 28426013 | Naufal Izza Fikry |

Folder ini berisi GA-OX, GA-PMX, dan random-key PSO untuk MC-TTRP. Permutasi berisi ID pelanggan, kemudian decoder membaginya menjadi rute. Semua metode meminimalkan total jarak. Skor pencarian menambahkan penalti kelebihan armada. Validator menghitung ulang biaya dan memeriksa solusi akhir.

## Instalasi dan perintah utama

Jalankan seluruh perintah berikut dari **root repo**, yaitu folder yang berisi `ga_pso/` dan `its_mcttrp/`.

```bash
bash setup.sh
.venv/bin/python reproduce.py
```

Dua perintah tersebut memasang dependensi dan memulihkan seluruh tabel serta gambar final laporan yang dipakai sebagai sumber presentasi. Tidak ada optimizer yang dijalankan pada mode ini. Panduan lengkap tersedia di [README root](../README.md).

Untuk Python saja: `python3 -m venv .venv`, kemudian `.venv/bin/python -m pip install -r ga_pso/requirements.txt`.

## Menjalankan GA dan PSO

```bash
.venv/bin/python ga_pso/run_experiments.py --only P1.csv
```

Runner menjalankan GA-OX, GA-PMX, dan PSO untuk semua seed pada P1. Hapus `--only P1.csv` untuk seluruh 14 kasus. Gunakan `.venv/bin/python ga_pso/run_experiments.py --sensitivity` untuk sensitivitas C01. Waktu setiap run dibaca dari `results/raw/its/`. Hasil ITS harus tersedia terlebih dahulu.

Untuk alur lengkap dari awal, gunakan `.venv/bin/python reproduce.py --full --fresh`. Opsi ini mengarsipkan hasil lama. Runner biasa melewati JSON yang sudah ada. Seed, armada, dan seluruh parameter berada dalam [config.json](config.json). Perubahan konfigurasi memerlukan hasil baru agar tidak tercampur dengan run lama.

## Hasil dan analisis

| Lokasi | Isi |
| --- | --- |
| `results/raw/` | JSON per algoritma, kasus, dan seed, termasuk ITS |
| `results/final_assets/` | Salinan tetap tabel dan gambar final laporan |
| `results/*.csv` | Ringkasan, reproduksi ITS, statistik, sensitivitas, dan seluruh run |
| `results/figures/` | Grafik analisis dalam PNG/PDF |
| `results/tables/` | Tabel LaTeX dari analisis |
| `data/` | Salinan data asli, termasuk checksum |
| `mcttrp/` | Parser, decoder, validator, dan pembaca hasil ITS |
| `tests/` | Pengujian operator, data dan alur kode |

`.venv/bin/python reproduce.py --analyze` menghitung ulang CSV, tabel, dan grafik dari JSON yang ada, kemudian memperbarui tabel dan tautan gambar pada README root. Mode default `reproduce.py` memakai aset final agar tampilan dan angka tetap sama dengan laporan yang menjadi sumber Figma.

## Batas model

Decoder menggunakan split dynamic programming. Trailer diparkir pada VC di main tour. Decoder membatasi satu sub-tour per MVR dan menempatkan VC pada main tour, sehingga ruang pencariannya lebih sempit daripada ITS. Batas armada dan aturan parkir mengikuti heuristik yang digunakan dalam eksperimen. PSO mengurutkan nilai random key untuk memperoleh urutan pelanggan. Inersia turun berdasarkan waktu yang terpakai.

## Perangkat dan batas reproduksi

Eksperimen yang dilaporkan menggunakan **Apple M5, CPU 10 inti, RAM 16 GB**, dengan 8 proses paralel. GA/PSO menggunakan Python 3.14 dan Numba, sedangkan ITS menggunakan R dan C++. Dependensi Python dicatat dalam `ga_pso/requirements.txt`.

Runtime dipengaruhi perangkat, jumlah proses yang berjalan bersamaan, dan implementasi. GA/PSO berhenti berdasarkan waktu, sehingga jumlah evaluasi dan hasil akhirnya dapat berbeda pada run ulang walaupun seed sama. Seed membantu mengendalikan bilangan acak, tetapi tidak menjamin hasil identik pada pencarian berbatas waktu. Nilai runtime paper berasal dari lingkungan penulis paper.

## Sumber

Paper acuan: Davila-Pena, Penas, dan Casas-Méndez (2023), [A new two-phase heuristic for a problem of food distribution with compartmentalized trucks and trailers](https://doi.org/10.1111/itor.13071).

Kode dan data ITS berasal dari [LauraDavilaPena/ITS_MC-TTRP](https://github.com/LauraDavilaPena/ITS_MC-TTRP), versi 0.1, berlisensi GPLv3 menurut `its_mcttrp/DESCRIPTION`. Atribusi dan README asli disimpan di `its_mcttrp/README.upstream.md`. Perbaikan lokal ditandai `[fix]` pada sumber terkait.
