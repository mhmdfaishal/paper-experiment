from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FS_FAI_METRICS_PATH, RESULTS_DIR, ROOT, summarize_experiments


CLASSIFIERS = ["Naive Bayes", "Decision Tree", "Logistic Regression"]
METHODS = [
    ("Paper", "#24476B"),
    ("Reproduce Paper", "#2A9D8F"),
    ("GA", "#E28E2C"),
    ("BPSO", "#8B6BB1"),
]
PUBLISHED_ACCURACY = {
    "Naive Bayes": 95.64,
    "Decision Tree": 95.58,
    "Logistic Regression": 94.88,
}
PUBLISHED_FEATURES = 39


def main() -> None:
    selected_runs = []
    for folder, algorithm in [("ga", "GA"), ("pso", "BPSO")]:
        runs = pd.read_csv(ROOT / folder / "results" / "experiment_runs.csv")
        validation = runs[runs.klasifier == "Logistic Regression"]
        best_method = validation.groupby("metode", sort=False)["fitness_validasi"].mean().idxmax()
        best_runs = runs[runs.metode == best_method].copy()
        best_runs["metode"] = algorithm
        selected_runs.append(best_runs)
    summary = summarize_experiments(pd.concat(selected_runs, ignore_index=True))
    paper_metrics = pd.read_csv(FS_FAI_METRICS_PATH)
    local_paper = paper_metrics[
        (paper_metrics["skenario"] == "Usulan paper")
        & (paper_metrics["parameter"] == 0.002)
    ].set_index("klasifier")
    summary = summary.set_index(["metode", "klasifier"])

    colors = [color for _, color in METHODS]
    positions = np.arange(len(CLASSIFIERS))
    width = 0.19
    fig, (accuracy_ax, feature_ax) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.8, 1]}
    )

    for method_index, (method, color) in enumerate(METHODS):
        offset = (method_index - 1.5) * width
        if method == "Paper":
            values = [PUBLISHED_ACCURACY[name] for name in CLASSIFIERS]
            errors = None
        elif method == "Reproduce Paper":
            values = [float(local_paper.loc[name, "akurasi_persen"]) for name in CLASSIFIERS]
            errors = None
        else:
            values = [
                float(summary.loc[(method.split()[0], name), "akurasi_mean"]) * 100
                for name in CLASSIFIERS
            ]
            errors = [
                float(summary.loc[(method.split()[0], name), "akurasi_std"]) * 100
                for name in CLASSIFIERS
            ]
        bars = accuracy_ax.bar(
            positions + offset, values, width, label=method, color=color,
            yerr=errors, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
        )
        for bar, value in zip(bars, values):
            accuracy_ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=8, rotation=90 if value < 40 else 0,
            )

    accuracy_ax.set_title("Akurasi pada test set")
    accuracy_ax.set_ylabel("Akurasi (%)")
    accuracy_ax.set_ylim(0, 108)
    accuracy_ax.set_xticks(positions, CLASSIFIERS)
    accuracy_ax.grid(axis="y", alpha=0.25)
    accuracy_ax.set_axisbelow(True)

    feature_counts = [
        PUBLISHED_FEATURES,
        int(local_paper["jumlah_fitur"].iloc[0]),
        float(summary.loc[("GA", "Naive Bayes"), "jumlah_fitur_mean"]),
        float(summary.loc[("BPSO", "Naive Bayes"), "jumlah_fitur_mean"]),
    ]
    feature_labels = ["Paper", "Reproduce Paper", "GA", "BPSO"]
    bars = feature_ax.bar(feature_labels, feature_counts, color=colors, width=0.66)
    for bar, value in zip(bars, feature_counts):
        feature_ax.annotate(
            f"{value:.1f}" if value % 1 else f"{int(value)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10,
        )
    feature_ax.set_title("Jumlah fitur yang dipertahankan")
    feature_ax.set_ylabel("Fitur (dari 543 awal)")
    feature_ax.set_ylim(0, max(feature_counts) * 1.2)
    feature_ax.grid(axis="y", alpha=0.25)
    feature_ax.set_axisbelow(True)

    fig.suptitle("Perbandingan metode usulan Paper, GA, dan BPSO", fontsize=16, y=0.98)
    handles, labels = accuracy_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.89))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "comparison_illustration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(RESULTS_DIR / "comparison_illustration.png")


if __name__ == "__main__":
    main()
