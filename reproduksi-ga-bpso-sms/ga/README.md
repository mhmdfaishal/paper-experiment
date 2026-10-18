# Genetic Algorithm

Program [ga_feature_selection.py](ga_feature_selection.py) memilih fitur kata menggunakan GA. Setiap kromosom berisi nilai 0 dan 1 untuk menentukan fitur yang dipakai.

## Jalankan Program

Jalankan dari folder `reproduksi-ga-bpso-sms`:

```bash
python ga/ga_feature_selection.py
```

Program membandingkan one-point dan two-point crossover pada lima seed. Pengaturan bawaan memakai 16 kromosom dan 15 generasi, total 240 evaluasi fitness per percobaan.

Untuk satu konfigurasi:

```bash
python ga/ga_feature_selection.py --seed 42 --crossover one-point
```

## Hasil

Performa tiap percobaan dan ringkasan ditampilkan di terminal. Hasil disimpan di `results/runs/`, `results/experiment_runs.csv`, dan `results/experiment_config.json`.
