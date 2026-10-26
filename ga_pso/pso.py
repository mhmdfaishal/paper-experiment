"""Particle swarm optimization with random-key encoding."""
import time

import numpy as np


def keys_to_perm(x):
    """Turn random keys into a permutation of customers 1..n."""
    return np.argsort(x, kind="stable") + 1


def inertia(frac, w_start, w_end):
    """Linearly decreasing inertia weight."""
    frac = min(max(frac, 0.0), 1.0)
    return w_start + (w_end - w_start) * frac


def update_particle(x, v, pbest, gbest, w, c1, c2, v_max, rng):
    """One velocity and position update."""
    n = x.shape[0]
    r1 = rng.random(n)
    r2 = rng.random(n)
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    v = np.clip(v, -v_max, v_max)
    x = np.clip(x + v, 0.0, 1.0)
    return x, v


def run_pso(decoder, seed, time_limit_s=None, max_evals=None, swarm_size=50,
            w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, v_max=0.2):
    """Run global-best PSO until the time or evaluation budget is used up."""
    if time_limit_s is None and max_evals is None:
        raise ValueError("give time_limit_s and/or max_evals")
    if max_evals is not None and max_evals < 1:
        raise ValueError("max_evals must be >= 1")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = decoder.inst.n

    X = rng.random((swarm_size, n))
    V = rng.uniform(-v_max, v_max, (swarm_size, n))
    pbest_x = X.copy()
    pbest_f = np.full(swarm_size, np.inf)
    gbest_x = X[0].copy()
    gbest_f = np.inf
    gbest_perm = None
    feas_f = np.inf
    feas_perm = None
    n_evals = 0
    iterations = 0
    tr_time, tr_evals, tr_best = [], [], []

    def out_of_budget():
        if max_evals is not None and n_evals >= max_evals:
            return True
        return time_limit_s is not None and time.perf_counter() - t0 >= time_limit_s

    def used_fraction():
        f = 0.0
        if max_evals is not None:
            f = n_evals / max_evals
        if time_limit_s is not None:
            f = max(f, (time.perf_counter() - t0) / time_limit_s)
        return f

    def trace():
        tr_time.append(time.perf_counter() - t0)
        tr_evals.append(n_evals)
        tr_best.append(float(feas_f))

    def evaluate(i):
        nonlocal n_evals, gbest_f, gbest_x, gbest_perm, feas_f, feas_perm
        perm = keys_to_perm(X[i])
        cost, feasible = decoder.evaluate(perm)
        n_evals += 1
        if cost < pbest_f[i]:
            pbest_f[i] = cost
            pbest_x[i] = X[i]
        if cost < gbest_f:
            gbest_f = cost
            gbest_x = X[i].copy()
            gbest_perm = perm
        if feasible and cost < feas_f:
            feas_f = cost
            feas_perm = perm

    done = True
    for i in range(swarm_size):
        if out_of_budget():
            done = False
            break
        evaluate(i)
    if done:
        iterations += 1
        trace()

    # gbest is updated right after each particle
    while done and not out_of_budget():
        for i in range(swarm_size):
            if out_of_budget():
                done = False
                break
            w = inertia(used_fraction(), w_start, w_end)
            X[i], V[i] = update_particle(X[i], V[i], pbest_x[i], gbest_x,
                                         w, c1, c2, v_max, rng)
            evaluate(i)
        else:
            iterations += 1
            trace()

    trace()
    if feas_perm is not None:
        best_cost, best_perm, feasible = float(feas_f), feas_perm, True
    else:
        best_cost, best_perm, feasible = float(gbest_f), gbest_perm, False
    return dict(best_cost=best_cost, best_perm=best_perm, feasible=feasible,
                n_evals=n_evals, iterations=iterations, trace_time=tr_time,
                trace_evals=tr_evals, trace_best=tr_best)
