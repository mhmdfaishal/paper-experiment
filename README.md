# Tugas Besar 1: Problem Solving Agent

**Mata kuliah:** AI5103 Artificial Intelligence and Data Science Fundamental

## Anggota Kelompok

| NIM | Nama |
| --- | --- |
| 28426006 | Muhammad Faishal Dienul Haq |
| 28426013 | Naufal Izza Fikry |

Proyek ini membandingkan Iterated Tabu Search (ITS), Genetic Algorithm dengan Order Crossover (GA-OX), Genetic Algorithm dengan Partially Mapped Crossover (GA-PMX), dan Particle Swarm Optimization (PSO) untuk rute truk dan trailer berkompartemen. Tujuannya meminimalkan total jarak sambil memenuhi permintaan, kapasitas kompartemen, akses pelanggan, dan batas armada.

## Struktur

| Lokasi | Isi |
| --- | --- |
| `ga_pso/` | Implementasi GA/PSO, decoder, validator, konfigurasi bersama, data dan hasil tersimpan |
| `its_mcttrp/` | Implementasi R/C++ ITS dari penulis paper dengan perbaikan lokal |
| `setup.sh` | Instalasi dependensi Python/R dan kompilasi pustaka C++ |
| `reproduce.py` | Menyiapkan kembali aset laporan atau menjalankan alur eksperimen opsional |
| `run_its_batch.py` | Batch ITS berdasarkan konfigurasi bersama |

## Instalasi

Clone repo lalu masuk ke foldernya. Sediakan Python 3.14, R 4.3 atau lebih baru, compiler C++, dan Bash. Pada macOS, pasang R terlebih dahulu, misalnya melalui Homebrew. Compiler tersedia melalui Xcode Command Line Tools (`xcode-select --install`). Linux membutuhkan toolchain C++ dan header pengembangan R. Windows dapat memakai WSL dengan lingkungan Linux.

Dari root repo, jalankan:

```bash
bash setup.sh
```

Script membuat `.venv`, memasang paket Python yang dipatok versinya, memasang Rcpp jika belum tersedia, lalu membangun `its_mcttrp/src/itsmcttrp.so`. Instalasi membutuhkan internet. Kegagalan satu tahap menghentikan script.

## Reproduksi tabel dan gambar final dalam satu perintah

```bash
.venv/bin/python reproduce.py
```

Perintah ini menyalin aset final yang disimpan di `ga_pso/results/final_assets/` ke `ga_pso/results/figures/`, kemudian memperbarui tabel dan tautan gambar di README root. Aset berasal dari laporan eksperimen 2 yang digunakan sebagai sumber konten presentasi Figma. Perintah ini mempertahankan tabel dan grafik final, tanpa menjalankan optimizer atau menghitung ulang hasil. Bahkan tanpa instalasi paket analisis, perintah ini dapat dijalankan dengan `python3 reproduce.py` karena hanya memakai pustaka standar.

Tabel ditampilkan dalam Markdown dari CSV hasil tersimpan. Empat gambar PNG tersedia: `gap_by_instance`, `convergence`, `routes_C01`, dan `sensitivity`. File LaTeX (`.tex` dan `.text`) serta PDF diabaikan oleh Git. Mode analisis dapat menghasilkan file tersebut secara lokal. Tata letak dan teks tambahan pada Figma tetap dikelola di Figma.

## Menjalankan eksperimen dari awal (opsional)

```bash
.venv/bin/python reproduce.py --full --fresh
```

Perintah tersebut mengarsipkan hasil mentah lama ke `ga_pso/results/raw_backup_<waktu>/`, menjalankan ITS, GA-OX, GA-PMX, PSO, sensitivitas, lalu membangun tabel dan gambar dari hasil baru. Aset final di `final_assets/` tetap disimpan. Seluruh alur dapat memerlukan beberapa jam. **Eksperimen tidak dijalankan ulang saat pembaruan dokumentasi ini.**

Gunakan `--full` tanpa `--fresh` untuk melanjutkan batch yang terputus. Hasil yang sudah memiliki JSON dilewati. Setelah mengubah konfigurasi, gunakan `--fresh` agar hasil dengan konfigurasi lama tidak tercampur. Untuk menghitung ulang analisis dari JSON yang ada, gunakan `.venv/bin/python reproduce.py --analyze`. Angka dan gambar yang dihitung ulang dapat berbeda dari aset final, terutama bila hasil mentah sudah berubah.

## Seed, parameter, dan cakupan

Konfigurasi bersama berada di `ga_pso/config.json`, termasuk parameter ITS. Kedua runner membacanya langsung.

| Bagian konfigurasi | Pengaturan |
| --- | --- |
| `instances` | P1 sampai P8 dan C01 sampai C06, jumlah truk/trailer serta nilai acuan |
| `seeds` | 1 sampai 10 untuk eksperimen utama |
| `workers` | 8 proses paralel |
| `its` | 100 iterasi, batas pengaman 36.000 detik |
| `ga` | Populasi 100, turnamen 3, crossover 0,9, mutasi inversi 0,2, elitisme 2, varian OX/PMX |
| `pso` | 50 partikel, inersia 0,9 menuju 0,4, c1=c2=2, kecepatan maksimum 0,2 |
| `sensitivity` | C01, seed 1 sampai 5, empat peluang mutasi GA dan empat inersia akhir PSO |
| `experiment_environment` | Perangkat eksperimen dan catatan runtime |

Anggaran GA/PSO adalah rata-rata waktu ITS tersimpan untuk kasus yang sama. Inersia PSO turun linear berdasarkan proporsi waktu yang terpakai. Data P berasal dari distribusi pakan koperasi, sedangkan data C adalah benchmark Chao yang diadaptasi ke MC-TTRP. Seluruh hasil utama mencakup 14 kasus dan 560 solusi akhir. Analisis sensitivitas disimpan terpisah. File data kedua implementasi diperiksa oleh test checksum.

## Pengujian

```bash
.venv/bin/python -m pytest ga_pso/tests -q
```

Pengujian mencakup parser, kelayakan solusi, decoder, operator GA/PSO, runner, dan kesamaan dataset. Pemeriksaan ini berbeda dari menjalankan ulang eksperimen penuh.

## Perangkat dan batas reproduksi

Eksperimen yang dilaporkan menggunakan **Apple M5, CPU 10 inti, RAM 16 GB**, dengan 8 proses paralel. GA/PSO menggunakan Python 3.14 dan Numba, sedangkan ITS menggunakan R dan C++. Dependensi Python dicatat dalam `ga_pso/requirements.txt`.

Runtime dipengaruhi perangkat, jumlah proses yang berjalan bersamaan, dan implementasi. GA/PSO berhenti berdasarkan waktu, sehingga jumlah evaluasi dan hasil akhirnya dapat berbeda pada run ulang walaupun seed sama. Seed membantu mengendalikan bilangan acak, tetapi tidak menjamin hasil identik pada pencarian berbatas waktu. Nilai runtime paper berasal dari lingkungan penulis paper.

## Sumber

Paper acuan: Davila-Pena, Penas, dan Casas-Méndez (2023), [A new two-phase heuristic for a problem of food distribution with compartmentalized trucks and trailers](https://doi.org/10.1111/itor.13071).

Kode dan data ITS berasal dari [LauraDavilaPena/ITS_MC-TTRP](https://github.com/LauraDavilaPena/ITS_MC-TTRP), versi 0.1, berlisensi GPLv3 menurut `its_mcttrp/DESCRIPTION`. Atribusi dan README asli disimpan di `its_mcttrp/README.upstream.md`. Perbaikan lokal ditandai `[fix]` pada sumber terkait.

<!-- results:start -->

## Hasil eksperimen

Angka berikut berasal dari hasil tersimpan. Biaya adalah total jarak. SD adalah simpangan baku. Gap dihitung terhadap acuan pada konfigurasi, dalam persen. Waktu dinyatakan dalam detik.

### Perbandingan ITS, GA-OX, GA-PMX, dan PSO

| Kasus | Metode | Terbaik | Rata-rata | SD | Terburuk | Gap rata-rata (%) | Waktu (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | ITS | 189,00 | 189,00 | 0,00 | 189,00 | 0,00 | 7,49 |
| P1 | GA-OX | 189,00 | 189,00 | 0,00 | 189,00 | 0,00 | 7,49 |
| P1 | GA-PMX | 189,00 | 189,00 | 0,00 | 189,00 | 0,00 | 7,49 |
| P1 | PSO | 189,00 | 189,00 | 0,00 | 189,00 | 0,00 | 7,49 |
| P2 | ITS | 140,00 | 141,40 | 0,97 | 142,00 | 1,00 | 7,30 |
| P2 | GA-OX | 140,00 | 140,00 | 0,00 | 140,00 | 0,00 | 7,30 |
| P2 | GA-PMX | 140,00 | 140,00 | 0,00 | 140,00 | 0,00 | 7,30 |
| P2 | PSO | 140,00 | 140,00 | 0,00 | 140,00 | 0,00 | 7,30 |
| P3 | ITS | 106,00 | 106,00 | 0,00 | 106,00 | 0,00 | 6,98 |
| P3 | GA-OX | 106,00 | 106,00 | 0,00 | 106,00 | 0,00 | 6,98 |
| P3 | GA-PMX | 106,00 | 106,00 | 0,00 | 106,00 | 0,00 | 6,98 |
| P3 | PSO | 106,00 | 106,00 | 0,00 | 106,00 | 0,00 | 6,98 |
| P4 | ITS | 256,00 | 256,00 | 0,00 | 256,00 | 0,00 | 8,44 |
| P4 | GA-OX | 256,00 | 256,00 | 0,00 | 256,00 | 0,00 | 8,44 |
| P4 | GA-PMX | 256,00 | 256,00 | 0,00 | 256,00 | 0,00 | 8,44 |
| P4 | PSO | 256,00 | 256,20 | 0,63 | 258,00 | 0,08 | 8,44 |
| P5 | ITS | 109,00 | 109,00 | 0,00 | 109,00 | 0,00 | 8,02 |
| P5 | GA-OX | 109,00 | 109,00 | 0,00 | 109,00 | 0,00 | 8,02 |
| P5 | GA-PMX | 109,00 | 109,00 | 0,00 | 109,00 | 0,00 | 8,02 |
| P5 | PSO | 109,00 | 109,40 | 0,84 | 111,00 | 0,37 | 8,02 |
| P6 | ITS | 222,00 | 224,10 | 1,45 | 225,00 | 0,95 | 9,40 |
| P6 | GA-OX | 225,00 | 225,00 | 0,00 | 225,00 | 1,35 | 9,40 |
| P6 | GA-PMX | 225,00 | 225,00 | 0,00 | 225,00 | 1,35 | 9,40 |
| P6 | PSO | 225,00 | 225,50 | 1,58 | 230,00 | 1,58 | 9,40 |
| P7 | ITS | 237,00 | 237,00 | 0,00 | 237,00 | 0,00 | 9,18 |
| P7 | GA-OX | 237,00 | 237,00 | 0,00 | 237,00 | 0,00 | 9,18 |
| P7 | GA-PMX | 237,00 | 237,00 | 0,00 | 237,00 | 0,00 | 9,18 |
| P7 | PSO | 237,00 | 237,10 | 0,32 | 238,00 | 0,04 | 9,18 |
| P8 | ITS | 207,00 | 212,30 | 1,89 | 213,00 | 2,56 | 7,80 |
| P8 | GA-OX | 207,00 | 207,00 | 0,00 | 207,00 | 0,00 | 7,81 |
| P8 | GA-PMX | 207,00 | 207,00 | 0,00 | 207,00 | 0,00 | 7,80 |
| P8 | PSO | 207,00 | 207,60 | 1,90 | 213,00 | 0,29 | 7,80 |
| C01 | ITS | 633,61 | 640,08 | 6,29 | 652,55 | -0,09 | 307,93 |
| C01 | GA-OX | 628,86 | 655,27 | 27,72 | 709,89 | 2,28 | 307,94 |
| C01 | GA-PMX | 652,34 | 675,34 | 16,46 | 702,41 | 5,41 | 307,94 |
| C01 | PSO | 745,40 | 808,76 | 28,99 | 845,46 | 26,24 | 307,94 |
| C02 | ITS | 711,21 | 723,55 | 8,26 | 736,85 | 2,18 | 263,60 |
| C02 | GA-OX | 718,06 | 745,56 | 16,57 | 761,21 | 5,29 | 263,60 |
| C02 | GA-PMX | 714,11 | 753,68 | 22,97 | 782,57 | 6,44 | 263,60 |
| C02 | PSO | 784,57 | 879,00 | 63,14 | 953,35 | 24,14 | 263,60 |
| C03 | ITS | 761,10 | 777,35 | 14,21 | 810,20 | 1,64 | 277,83 |
| C03 | GA-OX | 749,23 | 787,63 | 27,16 | 834,19 | 2,98 | 277,85 |
| C03 | GA-PMX | 759,32 | 808,55 | 35,62 | 888,85 | 5,72 | 277,83 |
| C03 | PSO | 860,59 | 925,44 | 55,45 | 1014,31 | 21,00 | 277,83 |
| C04 | ITS | 904,39 | 934,80 | 28,23 | 990,55 | 2,89 | 629,88 |
| C04 | GA-OX | 911,46 | 961,26 | 30,20 | 1005,77 | 5,80 | 629,88 |
| C04 | GA-PMX | 914,67 | 976,84 | 43,26 | 1075,41 | 7,52 | 629,88 |
| C04 | PSO | 1210,97 | 1277,36 | 42,68 | 1347,94 | 40,60 | 629,88 |
| C05 | ITS | 1012,91 | 1029,39 | 10,36 | 1046,69 | 1,63 | 541,75 |
| C05 | GA-OX | 1031,38 | 1083,65 | 40,15 | 1145,32 | 6,99 | 541,75 |
| C05 | GA-PMX | 1027,48 | 1078,55 | 43,90 | 1146,63 | 6,49 | 541,76 |
| C05 | PSO | 1217,55 | 1326,97 | 64,59 | 1433,72 | 31,02 | 541,75 |
| C06 | ITS | 1104,10 | 1143,67 | 18,70 | 1173,84 | 3,79 | 585,85 |
| C06 | GA-OX | 1155,94 | 1208,44 | 49,71 | 1299,82 | 9,67 | 585,85 |
| C06 | GA-PMX | 1154,91 | 1194,90 | 33,81 | 1250,44 | 8,44 | 585,85 |
| C06 | PSO | 1444,51 | 1507,57 | 44,09 | 1593,19 | 36,82 | 585,85 |

### Gap per kasus

![Gap per kasus](ga_pso/results/figures/gap_by_instance.png)

### Kurva konvergensi

![Kurva konvergensi](ga_pso/results/figures/convergence.png)

### Rute terbaik pada C01

![Rute terbaik ITS dan GA-OX pada C01](ga_pso/results/figures/routes_C01.png)

### Reproduksi ITS

CW adalah solusi awal Clarke-Wright. Waktu paper disajikan sesuai tabel sumber. Cara agregasi waktu pada paper tidak dirinci, sehingga perbandingan runtime perlu memperhatikan lingkungan pengujian.

| Kasus | CW paper | CW reproduksi | ITS paper | ITS terbaik | ITS rata-rata | Selisih terbaik (%) | Waktu paper (s) | Waktu reproduksi (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | n/a | 189,00 | 189,00 | 189,00 | 189,00 | 0,00 | 5,03 | 7,49 |
| P2 | n/a | 143,00 | 142,00 | 140,00 | 141,40 | -1,41 | 3,88 | 7,30 |
| P3 | n/a | 106,00 | 106,00 | 106,00 | 106,00 | 0,00 | 3,90 | 6,98 |
| P4 | n/a | 256,00 | 256,00 | 256,00 | 256,00 | 0,00 | 5,58 | 8,44 |
| P5 | n/a | 110,00 | 109,00 | 109,00 | 109,00 | 0,00 | 4,72 | 8,02 |
| P6 | n/a | 226,00 | 223,00 | 222,00 | 224,10 | -0,45 | 6,48 | 9,40 |
| P7 | n/a | 238,00 | 238,00 | 237,00 | 237,00 | -0,42 | 7,97 | 9,18 |
| P8 | n/a | 213,00 | 213,00 | 207,00 | 212,30 | -2,82 | 6,09 | 7,80 |
| C01 | 662,93 | 662,94 | 640,66 | 633,61 | 640,08 | -1,10 | 334,80 | 307,93 |
| C02 | 832,60 | 832,60 | 708,08 | 711,21 | 723,55 | 0,44 | 316,80 | 263,60 |
| C03 | 885,79 | 885,79 | 764,81 | 761,10 | 777,35 | -0,48 | 345,60 | 277,83 |
| C04 | 1013,57 | 1013,57 | 908,53 | 904,39 | 934,80 | -0,46 | 633,00 | 629,88 |
| C05 | 1168,01 | 1168,01 | 1012,83 | 1012,91 | 1029,39 | 0,01 | 535,80 | 541,75 |
| C06 | 1264,09 | 1264,09 | 1101,90 | 1104,10 | 1143,67 | 0,20 | 847,80 | 585,85 |

### Uji statistik

Uji Wilcoxon rank-sum membandingkan biaya antar metode. Kolom signifikan memakai ambang p < 0,05 tanpa koreksi perbandingan berganda.

| Kasus | Metode A | Metode B | p | Signifikan |
| --- | --- | --- | --- | --- |
| C01 | ITS | GA-OX | 0,496292 | False |
| C01 | GA-OX | GA-PMX | 0,069642 | False |
| C01 | GA-OX | PSO | 0,000157 | True |
| C02 | ITS | GA-OX | 0,006502 | True |
| C02 | GA-OX | GA-PMX | 0,364346 | False |
| C02 | GA-OX | PSO | 0,000157 | True |
| C03 | ITS | GA-OX | 0,596701 | False |
| C03 | GA-OX | GA-PMX | 0,226476 | False |
| C03 | GA-OX | PSO | 0,000157 | True |
| C04 | ITS | GA-OX | 0,069642 | False |
| C04 | GA-OX | GA-PMX | 0,496292 | False |
| C04 | GA-OX | PSO | 0,000157 | True |
| C05 | ITS | GA-PMX | 0,003197 | True |
| C05 | GA-OX | GA-PMX | 0,762369 | False |
| C05 | GA-PMX | PSO | 0,000157 | True |
| C06 | ITS | GA-PMX | 0,000670 | True |
| C06 | GA-OX | GA-PMX | 0,496292 | False |
| C06 | GA-PMX | PSO | 0,000157 | True |
| P1 | ITS | GA-OX | 1,000000 | False |
| P1 | GA-OX | GA-PMX | 1,000000 | False |
| P1 | GA-OX | PSO | 1,000000 | False |
| P2 | ITS | GA-OX | 0,008151 | True |
| P2 | GA-OX | GA-PMX | 1,000000 | False |
| P2 | GA-OX | PSO | 1,000000 | False |
| P3 | ITS | GA-OX | 1,000000 | False |
| P3 | GA-OX | GA-PMX | 1,000000 | False |
| P3 | GA-OX | PSO | 1,000000 | False |
| P4 | ITS | GA-OX | 1,000000 | False |
| P4 | GA-OX | GA-PMX | 1,000000 | False |
| P4 | GA-OX | PSO | 0,705457 | False |
| P5 | ITS | GA-OX | 1,000000 | False |
| P5 | GA-OX | GA-PMX | 1,000000 | False |
| P5 | GA-OX | PSO | 0,449692 | False |
| P6 | ITS | GA-OX | 0,256839 | False |
| P6 | GA-OX | GA-PMX | 1,000000 | False |
| P6 | GA-OX | PSO | 0,705457 | False |
| P7 | ITS | GA-OX | 1,000000 | False |
| P7 | GA-OX | GA-PMX | 1,000000 | False |
| P7 | GA-OX | PSO | 0,705457 | False |
| P8 | ITS | GA-OX | 0,000670 | True |
| P8 | GA-OX | GA-PMX | 1,000000 | False |
| P8 | GA-OX | PSO | 0,705457 | False |

### Sensitivitas parameter pada C01

mutation_rate adalah peluang mutasi GA. w_end adalah inersia akhir PSO. Masing-masing pengaturan memakai lima seed.

| Metode | Parameter | Nilai | Rata-rata | SD | Terbaik | Gap rata-rata (%) |
| --- | --- | --- | --- | --- | --- | --- |
| GA-OX | mutation_rate | 0,05 | 661,94 | 12,69 | 642,61 | 3,32 |
| GA-OX | mutation_rate | 0,10 | 672,25 | 25,32 | 640,73 | 4,93 |
| GA-OX | mutation_rate | 0,20 | 633,34 | 5,75 | 627,63 | -1,14 |
| GA-OX | mutation_rate | 0,40 | 633,78 | 19,59 | 605,58 | -1,07 |
| PSO | w_end | 0,20 | 803,56 | 18,73 | 776,04 | 25,43 |
| PSO | w_end | 0,40 | 813,38 | 40,81 | 776,53 | 26,96 |
| PSO | w_end | 0,60 | 827,44 | 28,71 | 788,00 | 29,16 |
| PSO | w_end | 0,80 | 861,67 | 41,76 | 796,95 | 34,50 |

![Sensitivitas parameter](ga_pso/results/figures/sensitivity.png)

Data lengkap tersedia pada [hasil mentah](ga_pso/results/raw/) dan [CSV seluruh run](ga_pso/results/all_runs.csv).

<!-- results:end -->
