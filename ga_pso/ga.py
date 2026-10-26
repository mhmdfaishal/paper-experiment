"""Genetic algorithm on the giant-tour permutation."""
import time

import numpy as np
from numba import njit


@njit(cache=True)
def order_crossover(p1, p2, a, b):
    """OX: keep p1[a..b], fill the rest in p2's order starting after b."""
    n = p1.shape[0]
    child = np.empty(n, np.int64)
    used = np.zeros(n + 1, np.bool_)
    for i in range(a, b + 1):
        child[i] = p1[i]
        used[p1[i]] = True
    pos = (b + 1) % n
    for k in range(n):
        v = p2[(b + 1 + k) % n]
        if not used[v]:
            child[pos] = v
            pos = (pos + 1) % n
    return child


@njit(cache=True)
def pmx_crossover(p1, p2, a, b):
    """PMX: keep p1[a..b], place the rest by the p1/p2 mapping."""
    n = p1.shape[0]
    child = np.full(n, -1, np.int64)
    in_seg = np.zeros(n + 1, np.bool_)
    pos2 = np.empty(n + 1, np.int64)
    for i in range(n):
        pos2[p2[i]] = i
    for i in range(a, b + 1):
        child[i] = p1[i]
        in_seg[p1[i]] = True
    for i in range(a, b + 1):
        v = p2[i]
        if in_seg[v]:
            continue
        pos = i
        while pos >= a and pos <= b:
            pos = pos2[p1[pos]]
        child[pos] = v
    for i in range(n):
        if child[i] < 0:
            child[i] = p2[i]
    return child


@njit(cache=True)
def inversion_mutation(perm, a, b):
    """Reverse perm[a..b]."""
    out = perm.copy()
    i, j = a, b
    while i < j:
        out[i], out[j] = perm[j], perm[i]
        i += 1
        j -= 1
    return out


def run_ga(decoder, seed, time_limit_s=None, max_evals=None, crossover="OX",
           pop_size=100, tournament_k=3, crossover_rate=0.9, mutation_rate=0.2, elite=2):
    """Run the GA until the time or evaluation budget is used up."""
    if time_limit_s is None and max_evals is None:
        raise ValueError("give time_limit_s and/or max_evals")
    if max_evals is not None and max_evals < 1:
        raise ValueError("max_evals must be >= 1")
    try:
        cx = {"OX": order_crossover, "PMX": pmx_crossover}[crossover]
    except KeyError:
        raise ValueError(f"unknown crossover {crossover!r}") from None
    if not 0 <= elite < pop_size:
        raise ValueError("need 0 <= elite < pop_size")

    t0 = time.perf_counter()
    n0 = decoder.n_evals
    n = decoder.inst.n
    rng = np.random.default_rng(seed)

    best_f, best_f_perm = np.inf, None      # best feasible
    best_a, best_a_perm = np.inf, None      # best including penalised
    tr_time, tr_evals, tr_best = [], [], []

    def used():
        return decoder.n_evals - n0

    def evaluate(perm):
        nonlocal best_f, best_f_perm, best_a, best_a_perm
        cost, feas = decoder.evaluate(perm)
        if feas and cost < best_f:
            best_f, best_f_perm = cost, perm.copy()
        if cost < best_a:
            best_a, best_a_perm = cost, perm.copy()
        return cost

    def out_of_budget():
        if max_evals is not None and used() >= max_evals:
            return True
        return time_limit_s is not None and time.perf_counter() - t0 >= time_limit_s

    def snapshot():
        tr_time.append(time.perf_counter() - t0)
        tr_evals.append(used())
        tr_best.append(best_f)

    pop = np.empty((pop_size, n), np.int64)
    costs = np.full(pop_size, np.inf)
    base = np.arange(1, n + 1, dtype=np.int64)
    stopped = False
    for i in range(pop_size):
        pop[i] = rng.permutation(base)
        costs[i] = evaluate(pop[i])
        if out_of_budget():
            stopped = True
            break
    generations = 0
    snapshot()

    n_child = pop_size - elite
    while not stopped:
        tour = rng.integers(0, pop_size, size=(n_child, 2, tournament_k))
        winners = np.take_along_axis(
            tour, costs[tour].argmin(axis=2)[..., None], axis=2)[..., 0]
        do_cx = rng.random(n_child) < crossover_rate
        cx_cuts = np.sort(rng.integers(0, n, size=(n_child, 2)), axis=1)
        do_mut = rng.random(n_child) < mutation_rate
        mu_cuts = np.sort(rng.integers(0, n, size=(n_child, 2)), axis=1)

        order = np.argsort(costs, kind="stable")
        new_pop = np.empty_like(pop)
        new_costs = np.empty_like(costs)
        new_pop[:elite] = pop[order[:elite]]
        new_costs[:elite] = costs[order[:elite]]

        for c in range(n_child):
            p1, p2 = pop[winners[c, 0]], pop[winners[c, 1]]
            if do_cx[c]:
                child = cx(p1, p2, cx_cuts[c, 0], cx_cuts[c, 1])
            else:
                child = p1.copy()
            if do_mut[c]:
                child = inversion_mutation(child, mu_cuts[c, 0], mu_cuts[c, 1])
            new_pop[elite + c] = child
            new_costs[elite + c] = evaluate(child)
            if out_of_budget():
                stopped = True
                break
        if stopped:
            break
        pop, costs = new_pop, new_costs
        generations += 1
        snapshot()

    snapshot()
    feasible = best_f_perm is not None
    return dict(
        best_cost=float(best_f if feasible else best_a),
        best_perm=(best_f_perm if feasible else best_a_perm).copy(),
        feasible=feasible,
        n_evals=used(),
        generations=generations,
        trace_time=tr_time,
        trace_evals=tr_evals,
        trace_best=tr_best,
    )
