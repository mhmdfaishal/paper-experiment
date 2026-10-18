# Tugas Besar 1: Problem Solving Agent

**Mata kuliah:** AI5103 Artificial Intelligence and Data Science Fundamental

Proyek ini mempelajari seleksi fitur kata untuk klasifikasi SMS spam. Eksperimen mencakup reproduksi metode paper serta seleksi fitur menggunakan Genetic Algorithm (GA) dan Binary Particle Swarm Optimization (BPSO).

## Anggota Kelompok

| NIM | Nama |
| --- | --- |
| 28426006 | Muhammad Faishal Dienul Haq |
| 28426013 | Naufal Izza Fikry |

## Isi Proyek

| Folder | Isi |
| --- | --- |
| [reproduksi-frequent-correlated-sms](reproduksi-frequent-correlated-sms/) | Reproduksi metode paper menggunakan FS-FAI. |
| [reproduksi-ga-bpso-sms](reproduksi-ga-bpso-sms/) | Eksperimen GA dan BPSO, variasi parameter, dan evaluasi klasifikasi. |

Eksperimen menggunakan 5.572 SMS dari UCI SMS Spam Collection dan 543 fitur kata awal. Setiap program memiliki dataset lokal.

**Paper acuan:** [A high-quality feature selection method based on frequent and correlated items for text classification](https://doi.org/10.1007/s00500-023-08587-x), Heba Mamdouh Farghaly dan Tarek Abd El-Hafeez, 2023.

## Instalasi

Jalankan dari folder `paper-experiment`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproduksi-frequent-correlated-sms/requirements.txt -r reproduksi-ga-bpso-sms/requirements.txt
```

Pada Windows, aktifkan lingkungan dengan `.venv\Scripts\Activate.ps1`.

## Jalankan Program

```bash
python reproduksi-frequent-correlated-sms/reproduce.py
python reproduksi-ga-bpso-sms/ga/ga_feature_selection.py
python reproduksi-ga-bpso-sms/pso/bpso_feature_selection.py
```

## Hasil

Hasil ditampilkan di terminal dan disimpan di folder `results` masing-masing program. Petunjuk eksperimen GA dan BPSO tersedia di [README GA dan BPSO](reproduksi-ga-bpso-sms/README.md).
