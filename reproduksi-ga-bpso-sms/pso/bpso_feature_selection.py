from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    evaluate_selected_features, make_inner_split, prepare_outer_split,
    print_convergence, print_experiment_summary,
    print_parameter_summary, print_performance,
    save_experiment_results, save_run, score_mask,
)

SEEDS_DEFAULT = "11,42,73,101,202"
INERTIA_VALUES = (0.40, 0.70, 0.90) # inertia controls the contribution of the previous velocity


# keep at least one selected feature.
def ensure_nonempty(position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not np.any(position):
        position[rng.integers(0, len(position))] = 1
    return position


# select features with BPSO using velocity, personal best, and global best
def run_bpso(swarm_size: int = 16, iterations: int = 14, seed: int = 42, inertia: float = 0.70, cognitive: float = 1.50, social: float = 1.50, run_label: str | None = None):
    X_train, X_test, y_train, y_test, feature_names, dataset = prepare_outer_split()
    X_inner, X_validation, y_inner, y_validation = make_inner_split(X_train, y_train)
    feature_count = X_train.shape[1]
    rng = np.random.default_rng(seed)
    fitness_evaluations = 0

    positions = (rng.random((swarm_size, feature_count)) < 0.20).astype(np.uint8)
    for index in range(swarm_size):
        positions[index] = ensure_nonempty(positions[index], rng)
    velocities = rng.uniform(-1.0, 1.0, size=(swarm_size, feature_count))
    personal_best = positions.copy()
    personal_scores = np.full(swarm_size, -1.0, dtype=float)
    personal_f1 = np.zeros(swarm_size, dtype=float)
    for particle in range(swarm_size):
        personal_scores[particle], personal_f1[particle] = score_mask(
            positions[particle], X_inner, y_inner, X_validation, y_validation
        )
        fitness_evaluations += 1

    global_index = int(np.argmax(personal_scores))
    global_best = personal_best[global_index].copy()
    global_score = float(personal_scores[global_index])
    global_f1 = float(personal_f1[global_index])
    history = [{
        "iterasi": 0,
        "evaluasi_kumulatif": swarm_size,
        "fitness_terbaik": global_score,
        "f1_macro_validasi_terbaik": global_f1,
        "fitur_terpilih": int(np.sum(global_best)),
    }]

    for iteration in range(iterations):
        for particle in range(swarm_size):
            r_personal = rng.random(feature_count)
            r_social = rng.random(feature_count)
            velocities[particle] = (
                inertia * velocities[particle]
                + cognitive * r_personal
                * (personal_best[particle].astype(float) - positions[particle].astype(float))
                + social * r_social
                * (global_best.astype(float) - positions[particle].astype(float))
            )
            velocities[particle] = np.clip(velocities[particle], -6.0, 6.0)
            probabilities = expit(velocities[particle])
            positions[particle] = (rng.random(feature_count) < probabilities).astype(np.uint8)
            positions[particle] = ensure_nonempty(positions[particle], rng)
            current_score, current_f1 = score_mask(
                positions[particle], X_inner, y_inner, X_validation, y_validation
            )
            fitness_evaluations += 1
            if current_score > personal_scores[particle]:
                personal_best[particle] = positions[particle].copy()
                personal_scores[particle] = current_score
                personal_f1[particle] = current_f1

        global_index = int(np.argmax(personal_scores))
        if personal_scores[global_index] > global_score:
            global_best = personal_best[global_index].copy()
            global_score = float(personal_scores[global_index])
            global_f1 = float(personal_f1[global_index])
        history.append({
            "iterasi": iteration + 1,
            "evaluasi_kumulatif": (iteration + 2) * swarm_size,
            "fitness_terbaik": global_score,
            "f1_macro_validasi_terbaik": global_f1,
            "fitur_terpilih": int(np.sum(global_best)),
        })

    selected_indices = np.flatnonzero(global_best)
    model_rows = evaluate_selected_features(X_train, y_train, X_test, y_test, selected_indices)
    config = {
        "seed": seed,
        "run_label": run_label,
        "swarm_size": swarm_size,
        "iterations": iterations,
        "inertia_weight": inertia,
        "cognitive_coefficient": cognitive,
        "social_coefficient": social,
        "velocity_bounds": [-6.0, 6.0],
        "initial_feature_probability": 0.20,
        "objective": "0.50 * F1-macro-validasi + 0.50 * (1 - proporsi_fitur)",
        "validation_fraction_of_training": 0.20,
        "fitness_evaluations": fitness_evaluations,
    }
    save_run("bpso", config, history, feature_names, selected_indices, model_rows,
             dataset, X_train, y_train, X_test, y_test)
    return model_rows, history, selected_indices, config


# compare inertia values across seeds and print experiment results
def main():
    parser = argparse.ArgumentParser(description="Seleksi fitur SMS dengan Binary Particle Swarm Optimization.")
    parser.add_argument("--swarm-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=14,
                        help="Jumlah pembaruan setelah evaluasi partikel awal.")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seeds", default=SEEDS_DEFAULT)
    seed_options.add_argument("--seed", type=int, help="Jalankan satu seed saja.")
    parser.add_argument("--inertia", type=float, help="Jalankan satu nilai inertia saja.")
    args = parser.parse_args()
    try:
        seeds = [args.seed] if args.seed is not None else [int(value) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("Seed harus berupa angka yang dipisahkan koma.")
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("Seed harus berbeda dan bernilai nol atau lebih.")
    if args.seed is None and len(seeds) < 5:
        parser.error("Eksperimen memerlukan minimal lima seed. Gunakan --seed untuk satu run.")
    if args.swarm_size < 1 or args.iterations < 0:
        parser.error("Swarm minimal satu dan jumlah pembaruan minimal nol.")
    if args.inertia is not None and (not np.isfinite(args.inertia) or args.inertia < 0):
        parser.error("Inertia harus berupa angka terhingga bernilai nol atau lebih.")

    inertia_values = INERTIA_VALUES if args.inertia is None else (args.inertia,)
    X_train, X_test, y_train, y_test, feature_names, dataset = prepare_outer_split()
    evaluation_budget = args.swarm_size * (1 + args.iterations)
    print(f"Dataset: {len(dataset):,} SMS, {len(feature_names)} fitur.")
    print(f"Seed: {', '.join(str(seed) for seed in seeds)}")
    print(f"Budget per run: {args.swarm_size} partikel × (1 awal + {args.iterations} pembaruan) = {evaluation_budget} evaluasi.")

    rows = []
    histories = {}
    run_number = 0
    total_runs = len(inertia_values) * len(seeds)
    for inertia in inertia_values:
        method = f"BPSO inertia={inertia:.2f}"
        histories[method] = []
        for seed in seeds:
            run_number += 1
            print(f"\n[{run_number}/{total_runs}] {method}, seed={seed}", flush=True)
            model_rows, history, selected, config = run_bpso(
                args.swarm_size, args.iterations, seed, inertia=inertia,
                run_label=f"inertia_{inertia:.2f}_seed_{seed}",
            )
            histories[method].append(history)
            print(f"Fitur: {len(selected)}/{len(feature_names)}, fitness validasi: {history[-1]['fitness_terbaik']:.4f}")
            print(f"Evaluasi fitness: {config['fitness_evaluations']}")
            print_performance(model_rows)
            for model_row in model_rows:
                rows.append({
                    "metode": method, "seed": seed, **model_row,
                    "fitness_validasi": history[-1]["fitness_terbaik"],
                    "evaluasi_fitness": config["fitness_evaluations"],
                })

    baseline = evaluate_selected_features(X_train, y_train, X_test, y_test, list(range(len(feature_names))))
    summary_rows = rows.copy()
    for model_row in baseline:
        summary_rows.append({"metode": "Tanpa seleksi", "seed": "tetap", **model_row,
                             "fitness_validasi": np.nan, "evaluasi_fitness": 0})
    print_experiment_summary(summary_rows)
    print_parameter_summary(rows, "Sensitivitas inertia BPSO")
    print_convergence(histories)
    if len(seeds) >= 5:
        save_experiment_results("bpso", rows, feature_names)
    print(f"\nData mentah per run: {Path(__file__).parent / 'results' / 'runs'}")


if __name__ == "__main__":
    main()
