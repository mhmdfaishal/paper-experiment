# Seleksi Fitur SMS dengan GA dan BPSO

Program ini memilih fitur kata untuk klasifikasi SMS spam dan ham menggunakan Genetic Algorithm (GA) dan Binary Particle Swarm Optimization (BPSO).

Dataset berisi 5.572 SMS dengan 543 fitur awal. Dataset tersedia di `data/raw/`. Daftar fitur awal dan hasil FS-FAI lokal tersedia di `data/reference/`.

## Instalasiu

Jalankan dari folder `reproduksi-ga-bpso-sms`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pada Windows, aktifkan lingkungan dengan `venv\Scripts\Activate.ps1`.

## Jalankan Eksperimen

```bash
python ga/ga_feature_selection.py
python pso/bpso_feature_selection.py
```

- GA membandingkan one-point dan two-point crossover pada lima seed, total 10 percobaan.
- BPSO membandingkan inertia 0.40, 0.70, dan 0.90 pada lima seed, total 15 percobaan.
- Setiap percobaan menggunakan 240 evaluasi fitness dengan pengaturan bawaan.

Untuk menjalankan satu konfigurasi:

```bash
python ga/ga_feature_selection.py --seed 42 --crossover one-point
python pso/bpso_feature_selection.py --seed 42 --inertia 0.70
```

## Hasil

Terminal menampilkan performa tiap percobaan, ringkasan lima seed, perbandingan parameter, konvergensi, dan perbandingan dengan FS-FAI lokal. Perbandingan statistik GA dan BPSO muncul setelah kedua eksperimen lima seed selesai.

Hasil tiap percobaan disimpan di `ga/results/runs/` dan `pso/results/runs/`. Ringkasan eksperimen disimpan sebagai `experiment_runs.csv` dan `experiment_config.json` di folder `results` masing-masing algoritma.
