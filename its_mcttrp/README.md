# Tugas Besar 1: Problem Solving Agent

**Mata kuliah:** AI5103 Artificial Intelligence and Data Science Fundamental

## Anggota Kelompok

| NIM | Nama |
| --- | --- |
| 28426006 | Muhammad Faishal Dienul Haq |
| 28426013 | Naufal Izza Fikry |

Folder ini memuat baseline Clarke-Wright dan Iterated Tabu Search (ITS) dari kode penulis paper MC-TTRP. Sumber asli dan atribusinya tersedia pada [README.upstream.md](README.upstream.md). `DESCRIPTION` menyatakan lisensi GPLv3.

## Instalasi

Dari root repo, jalankan:

```bash
bash setup.sh
```

Diperlukan R 4.3 atau lebih baru, Rcpp, compiler C++, Python 3.14, dan Bash. Script mengompilasi `src/movlist.cpp` menjadi pustaka untuk R. Instalasi manual khusus ITS dari root repo:

```bash
Rscript -e 'install.packages("Rcpp", repos="https://cloud.r-project.org")'
bash its_mcttrp/src/build.sh
```

## Menjalankan

Satu kasus, dijalankan dari folder `its_mcttrp/`:

```bash
Rscript run_its.R instances/P1.csv 2 1 100 600 1 /tmp/P1_s1.json
```

Urutan argumen: file kasus, jumlah truk, jumlah trailer, iterasi, batas waktu detik, seed, dan lokasi JSON keluaran. Contoh memakai dua truk, satu trailer, 100 iterasi, batas 600 detik, dan seed 1.

Batch berdasarkan konfigurasi bersama, dijalankan dari root repo:

```bash
.venv/bin/python run_its_batch.py --only P1.csv
```

Hilangkan `--only P1.csv` untuk semua 14 kasus dan seed 1 sampai 10. Parameter batch tersimpan di `ga_pso/config.json`, bagian `instances`, `seeds`, `workers`, dan `its`. Hasil ditulis ke `ga_pso/results/raw/its/`. JSON yang sudah ada dilewati. Seed diatur pada pemanggilan solver melalui `run_its.R`.

## Reproduksi tabel dan gambar laporan

Dari root repo:

```bash
.venv/bin/python reproduce.py
```

Perintah ini memulihkan tabel dan gambar final yang sudah digunakan dalam laporan sebagai sumber presentasi, tanpa menjalankan ulang ITS. Data hasil mentah ITS juga disertakan. Untuk alur eksperimen baru beserta analisisnya, gunakan `.venv/bin/python reproduce.py --full --fresh`. Penjelasan pengarsipan hasil lama tersedia di [README root](../README.md).

## Struktur dan perubahan lokal

`R/` berisi solver, `src/` berisi operator C++, `instances/` berisi data, `load_its.R` memuat solver, dan `run_its.R` menulis JSON hasil serta jejak konvergensi. Folder `reproducibility/` mempertahankan script asli penulis. Alur tugas ini memakai runner dan konfigurasi bersama karena sebagian script asli memakai jumlah iterasi berbeda dari 100 pada kasus besar.

Perbaikan lokal mencakup pemuatan pustaka lintas sistem operasi, kompatibilitas operator logika vektor pada R baru, indeks produk pada perhitungan penalti C++, pemuatan solver pada script asli, variabel hasil yang tidak terdefinisi, dan pencatatan konvergensi. Perubahan terkait ditandai `[fix]` pada kode. `README.upstream.md` mempertahankan dokumentasi penulis.

## Perangkat dan batas reproduksi

Eksperimen yang dilaporkan menggunakan **Apple M5, CPU 10 inti, RAM 16 GB**, dengan 8 proses paralel. GA/PSO menggunakan Python 3.14 dan Numba, sedangkan ITS menggunakan R dan C++. Dependensi Python dicatat dalam `ga_pso/requirements.txt`.

Runtime dipengaruhi perangkat, jumlah proses yang berjalan bersamaan, dan implementasi. GA/PSO berhenti berdasarkan waktu, sehingga jumlah evaluasi dan hasil akhirnya dapat berbeda pada run ulang walaupun seed sama. Seed membantu mengendalikan bilangan acak, tetapi tidak menjamin hasil identik pada pencarian berbatas waktu. Nilai runtime paper berasal dari lingkungan penulis paper.

## Sumber

Paper acuan: Davila-Pena, Penas, dan Casas-Méndez (2023), [A new two-phase heuristic for a problem of food distribution with compartmentalized trucks and trailers](https://doi.org/10.1111/itor.13071).

Kode dan data ITS berasal dari [LauraDavilaPena/ITS_MC-TTRP](https://github.com/LauraDavilaPena/ITS_MC-TTRP), versi 0.1, berlisensi GPLv3 menurut `its_mcttrp/DESCRIPTION`. Atribusi dan README asli disimpan di `its_mcttrp/README.upstream.md`. Perbaikan lokal ditandai `[fix]` pada sumber terkait.
