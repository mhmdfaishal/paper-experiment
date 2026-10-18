from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    evaluate_selected_features, make_inner_split, prepare_outer_split,
    print_convergence, print_experiment_summary,
    print_parameter_summary, print_performance,
    save_experiment_results, save_run, score_mask,
)

SEEDS_DEFAULT = "11,42,73,101,202"
CROSSOVER_METHODS = ("one-point", "two-point")


# keep at least one selected feature
def ensure_nonempty(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    if not np.any(mask):
        mask[rng.randrange(len(mask))] = 1
    return mask


# select the fittest parent from random tournament candidates
def tournament(population, fitness_values, rng, tournament_size=3):
    candidates = rng.sample(range(len(population)), min(tournament_size, len(population)))
    winner = candidates[0]
    for candidate in candidates[1:]:
        if fitness_values[candidate] > fitness_values[winner]:
            winner = candidate
    return population[winner].copy()


# create two children using one-point or two-point crossover
def crossover(parent_a: np.ndarray, parent_b: np.ndarray, method: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    feature_count = len(parent_a)

    if feature_count < 2:
        return child_a, child_b

    if method == "one-point":
        point = rng.randrange(1, feature_count)
        child_a[point:], child_b[point:] = parent_b[point:].copy(), parent_a[point:].copy()
    elif method == "two-point":
        if feature_count < 3:
            point = rng.randrange(1, feature_count)
            child_a[point:], child_b[point:] = parent_b[point:].copy(), parent_a[point:].copy()
        else:
            point_a, point_b = sorted(rng.sample(range(1, feature_count), 2))
            child_a[point_a:point_b], child_b[point_a:point_b] = (
                parent_b[point_a:point_b].copy(), parent_a[point_a:point_b].copy()
            )
    else:
        raise ValueError(f"Metode crossover tidak dikenal: {method}")

    return child_a, child_b


# select features with GA and evaluate the best solution on test data
def run_ga(population_size: int = 16, generations: int = 15, seed: int = 42, run_label: str | None = None, crossover_method: str = "one-point", crossover_probability: float = 0.80):
    X_train, X_test, y_train, y_test, feature_names, dataset = prepare_outer_split()
    X_inner, X_validation, y_inner, y_validation = make_inner_split(X_train, y_train)
    feature_count = X_train.shape[1]
    rng = random.Random(seed)
    fitness_evaluations = 0

    population = []
    for _ in range(population_size):
        mask = np.zeros(feature_count, dtype=np.uint8)
        for index in range(feature_count):
            if rng.random() < 0.20:
                mask[index] = 1
        population.append(ensure_nonempty(mask, rng))

    history = []
    best_mask = None
    best_fitness = -1.0
    best_f1 = 0.0
    for generation in range(generations):
        fitness_values = []
        f1_values = []
        for i in range(len(population)):
            fitness, f1_macro = score_mask(population[i], X_inner, y_inner, X_validation, y_validation)
            fitness_evaluations += 1
            fitness_values.append(fitness)
            f1_values.append(f1_macro)
        generation_best = int(np.argmax(fitness_values))
        if fitness_values[generation_best] > best_fitness:
            best_fitness = fitness_values[generation_best]
            best_f1 = f1_values[generation_best]
            best_mask = population[generation_best].copy()
        history.append({
            "generasi": generation + 1,
            "evaluasi_kumulatif": (generation + 1) * population_size,
            "fitness_terbaik": float(best_fitness),
            "f1_macro_validasi_terbaik": float(best_f1),
            "fitur_terpilih": int(np.sum(best_mask)),
        })

        ranked = sorted(range(population_size), key=lambda index: fitness_values[index], reverse=True)
        next_population = [population[ranked[0]].copy(), population[ranked[1]].copy()]
        while len(next_population) < population_size:
            parent_a = tournament(population, fitness_values, rng)
            parent_b = tournament(population, fitness_values, rng)
            child_a = parent_a.copy()
            child_b = parent_b.copy()
            if rng.random() < crossover_probability:
                child_a, child_b = crossover(parent_a, parent_b, crossover_method, rng)
            for child in [child_a, child_b]:
                for gene in range(feature_count):
                    if rng.random() < 1.0 / feature_count:
                        child[gene] = 1 - child[gene]
                next_population.append(ensure_nonempty(child, rng))
                if len(next_population) >= population_size:
                    break
        population = next_population[:population_size]

    selected_indices = np.flatnonzero(best_mask)
    model_rows = evaluate_selected_features(X_train, y_train, X_test, y_test, selected_indices)
    config = {
        "seed": seed,
        "run_label": run_label,
        "population_size": population_size,
        "generations": generations,
        "elitism": 2,
        "selection": "tournament-3",
        "crossover_method": crossover_method,
        "crossover_probability": crossover_probability,
        "mutation_probability_per_gene": 1.0 / feature_count,
        "initial_feature_probability": 0.20,
        "objective": "0.50 * F1-macro-validasi + 0.50 * (1 - proporsi_fitur)",
        "validation_fraction_of_training": 0.20,
        "fitness_evaluations": fitness_evaluations,
    }
    save_run("ga", config, history, feature_names, selected_indices, model_rows, dataset, X_train, y_train, X_test, y_test)
    return model_rows, history, selected_indices, config


# compare crossover methods across seeds and print experiment results.
def main():
    parser = argparse.ArgumentParser(description="Seleksi fitur SMS dengan Genetic Algorithm biner.")
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=15)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seeds", default=SEEDS_DEFAULT)
    seed_options.add_argument("--seed", type=int, help="Jalankan satu seed saja.")
    parser.add_argument("--crossover", choices=CROSSOVER_METHODS, help="Jalankan satu jenis crossover saja.")
    args = parser.parse_args()
    try:
        seeds = [args.seed] if args.seed is not None else [int(value) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("Seed harus berupa angka yang dipisahkan koma.")
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("Seed harus berbeda dan bernilai nol atau lebih.")
    if args.seed is None and len(seeds) < 5:
        parser.error("Eksperimen memerlukan minimal lima seed. Gunakan --seed untuk satu run.")
    if args.population < 2 or args.generations < 1:
        parser.error("Populasi minimal dua dan generasi minimal satu.")

    crossovers = CROSSOVER_METHODS if args.crossover is None else (args.crossover,)
    X_train, X_test, y_train, y_test, feature_names, dataset = prepare_outer_split()
    evaluation_budget = args.population * args.generations
    print(f"Dataset: {len(dataset):,} SMS, {len(feature_names)} fitur.")
    print(f"Seed: {', '.join(str(seed) for seed in seeds)}")
    print(f"Budget per run: {args.population} kandidat × {args.generations} generasi = {evaluation_budget} evaluasi.")

    rows = []
    histories = {}
    run_number = 0
    total_runs = len(crossovers) * len(seeds)
    for crossover_method in crossovers:
        method = f"GA {crossover_method}"
        histories[method] = []
        for seed in seeds:
            run_number += 1
            print(f"\n[{run_number}/{total_runs}] {method}, seed={seed}", flush=True)
            model_rows, history, selected, config = run_ga(
                args.population, args.generations, seed,
                run_label=f"{crossover_method.replace('-', '_')}_seed_{seed}",
                crossover_method=crossover_method,
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
        summary_rows.append({"metode": "Tanpa seleksi", "seed": "tetap", **model_row, "fitness_validasi": np.nan, "evaluasi_fitness": 0})
    print_experiment_summary(summary_rows)
    print_parameter_summary(rows, "Perbandingan crossover GA")
    print_convergence(histories)
    if len(seeds) >= 5:
        save_experiment_results("ga", rows, feature_names)
    print(f"\nData mentah per run: {Path(__file__).parent / 'results' / 'runs'}")


if __name__ == "__main__":
    main()
