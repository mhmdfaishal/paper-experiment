"""Render saved CSV tables and figure links in the root README without rerunning experiments."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def table(filename, columns):
    with (ROOT / "ga_pso/results" / filename).open() as stream:
        rows = list(csv.DictReader(stream))
    lines = ["| " + " | ".join(label for key, label in columns) + " |",
             "| " + " | ".join("---" for column in columns) + " |"]
    for row in rows:
        values = []
        for key, label in columns:
            value = row[key].replace("CHAO_MCTTRP_", "C")
            if value == "":
                value = "n/a"
            else:
                try:
                    number = float(value)
                    value = f"{number:.6f}" if key == "p_value" else f"{number:.2f}"
                    value = value.replace(".", ",")
                except ValueError:
                    pass
            values.append(value.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parts = ["## Hasil eksperimen", "Angka berikut berasal dari hasil tersimpan. Biaya adalah total jarak. SD adalah simpangan baku. Gap dihitung terhadap acuan pada konfigurasi, dalam persen. Waktu dinyatakan dalam detik."]
    parts += ["### Perbandingan ITS, GA-OX, GA-PMX, dan PSO", table("summary.csv", [("instance", "Kasus"), ("algorithm", "Metode"), ("best", "Terbaik"), ("mean", "Rata-rata"), ("std", "SD"), ("worst", "Terburuk"), ("gap_mean", "Gap rata-rata (%)"), ("time_mean", "Waktu (s)")])]
    parts += ["### Gap per kasus", "![Gap per kasus](ga_pso/results/figures/gap_by_instance.png)", "### Kurva konvergensi", "![Kurva konvergensi](ga_pso/results/figures/convergence.png)", "### Rute terbaik pada C01", "![Rute terbaik ITS dan GA-OX pada C01](ga_pso/results/figures/routes_C01.png)"]
    parts += ["### Reproduksi ITS", "CW adalah solusi awal Clarke-Wright. Waktu paper disajikan sesuai tabel sumber. Cara agregasi waktu pada paper tidak dirinci, sehingga perbandingan runtime perlu memperhatikan lingkungan pengujian.", table("reproduction.csv", [("instance", "Kasus"), ("paper_cw", "CW paper"), ("ours_cw", "CW reproduksi"), ("paper_its", "ITS paper"), ("ours_best", "ITS terbaik"), ("ours_mean", "ITS rata-rata"), ("diff_best_pct", "Selisih terbaik (%)"), ("paper_time_s", "Waktu paper (s)"), ("ours_time_s", "Waktu reproduksi (s)")])]
    parts += ["### Uji statistik", "Uji Wilcoxon rank-sum membandingkan biaya antar metode. Kolom signifikan memakai ambang p < 0,05 tanpa koreksi perbandingan berganda.", table("stats.csv", [("instance", "Kasus"), ("a", "Metode A"), ("b", "Metode B"), ("p_value", "p"), ("significant_0_05", "Signifikan")])]
    parts += ["### Sensitivitas parameter pada C01", "mutation_rate adalah peluang mutasi GA. w_end adalah inersia akhir PSO. Masing-masing pengaturan memakai lima seed.", table("sensitivity.csv", [("algorithm", "Metode"), ("param", "Parameter"), ("value", "Nilai"), ("mean", "Rata-rata"), ("std", "SD"), ("min", "Terbaik"), ("gap_mean", "Gap rata-rata (%)")]), "![Sensitivitas parameter](ga_pso/results/figures/sensitivity.png)", "Data lengkap tersedia pada [hasil mentah](ga_pso/results/raw/) dan [CSV seluruh run](ga_pso/results/all_runs.csv)."]
    path = ROOT / "README.md"
    text = path.read_text()
    block = START + "\n\n" + "\n\n".join(parts) + "\n\n" + END
    if START in text:
        first, rest = text.split(START, 1)
        _, last = rest.split(END, 1)
        text = first + block + last
    else:
        text += "\n" + block + "\n"
    path.write_text(text)


if __name__ == "__main__":
    main()
