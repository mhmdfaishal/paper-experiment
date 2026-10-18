# Binary Particle Swarm Optimization

Program [bpso_feature_selection.py](bpso_feature_selection.py) memilih fitur kata menggunakan BPSO. Setiap partikel berisi nilai 0 dan 1 untuk menentukan fitur yang dipakai.

## Jalankan Program

Jalankan dari folder `reproduksi-ga-bpso-sms`:

```bash
python pso/bpso_feature_selection.py
```

Program membandingkan inertia 0.40, 0.70, dan 0.90 pada lima seed. Pengaturan bawaan memakai 16 partikel dan 14 pembaruan. Evaluasi awal dan pembaruan menghasilkan 240 evaluasi fitness per percobaan.

Untuk satu konfigurasi:

```bash
python pso/bpso_feature_selection.py --seed 42 --inertia 0.70
```

## Hasil

Performa tiap percobaan, ringkasan, dan sensitivitas inertia ditampilkan di terminal. Hasil disimpan di `results/runs/`, `results/experiment_runs.csv`, dan `results/experiment_config.json`.
