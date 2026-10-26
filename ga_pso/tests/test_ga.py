import time
from pathlib import Path

import numpy as np
import pytest

from ga import inversion_mutation, order_crossover, pmx_crossover, run_ga
from mcttrp.decoder import Decoder
from mcttrp.instance import load_instance
from mcttrp.solution import solution_cost, validate

INST = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def p1():
    return load_instance(INST / "P1.csv", 2, 1)


@pytest.fixture(scope="module")
def chao():
    return load_instance(INST / "CHAO_MCTTRP_01.txt", 8, 5)


def _cuts(rng, n):
    a, b = np.sort(rng.integers(0, n, size=2))
    return int(a), int(b)


def _parents(rng, n):
    base = np.arange(1, n + 1, dtype=np.int64)
    return rng.permutation(base), rng.permutation(base)


@pytest.mark.parametrize("op", [order_crossover, pmx_crossover])
@pytest.mark.parametrize("n", [8, 50])
def test_crossover_returns_valid_permutation(op, n):
    rng = np.random.default_rng(0)
    for _ in range(500):
        p1_, p2_ = _parents(rng, n)
        a, b = _cuts(rng, n)
        child = op(p1_, p2_, a, b)
        assert sorted(child.tolist()) == list(range(1, n + 1))


@pytest.mark.parametrize("op", [order_crossover, pmx_crossover])
@pytest.mark.parametrize("n", [8, 50])
def test_crossover_keeps_parent1_segment_in_place(op, n):
    rng = np.random.default_rng(1)
    for _ in range(500):
        p1_, p2_ = _parents(rng, n)
        a, b = _cuts(rng, n)
        child = op(p1_, p2_, a, b)
        assert np.array_equal(child[a:b + 1], p1_[a:b + 1])


@pytest.mark.parametrize("op", [order_crossover, pmx_crossover])
def test_crossover_does_not_modify_parents(op):
    rng = np.random.default_rng(2)
    p1_, p2_ = _parents(rng, 20)
    c1, c2 = p1_.copy(), p2_.copy()
    op(p1_, p2_, 3, 9)
    assert np.array_equal(p1_, c1) and np.array_equal(p2_, c2)


def test_pmx_outside_segment_follows_parent2_when_no_conflict():
    p1_ = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    p2_ = np.array([3, 7, 5, 1, 6, 8, 2, 4], dtype=np.int64)
    child = pmx_crossover(p1_, p2_, 3, 5)  # p1 segment = [4, 5, 6]
    assert child.tolist() == [3, 7, 8, 4, 5, 6, 2, 1]


def test_ox_fills_from_parent2_order_after_segment():
    p1_ = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    p2_ = np.array([8, 6, 4, 2, 7, 5, 3, 1], dtype=np.int64)
    child = order_crossover(p1_, p2_, 2, 4)  # keeps [3, 4, 5]
    assert child.tolist() == [2, 7, 3, 4, 5, 1, 8, 6]


@pytest.mark.parametrize("n", [8, 50])
def test_inversion_mutation_valid_and_reverses_segment(n):
    rng = np.random.default_rng(3)
    for _ in range(300):
        p, _ = _parents(rng, n)
        a, b = _cuts(rng, n)
        orig = p.copy()
        out = inversion_mutation(p, a, b)
        assert sorted(out.tolist()) == list(range(1, n + 1))
        assert np.array_equal(out[a:b + 1], orig[a:b + 1][::-1])
        assert np.array_equal(out[:a], orig[:a]) and np.array_equal(out[b + 1:], orig[b + 1:])
        assert np.array_equal(p, orig)


@pytest.mark.parametrize("cx", ["OX", "PMX"])
def test_budget_respected_and_deterministic(p1, cx):
    runs = []
    for _ in range(2):
        dec = Decoder(p1)
        r = run_ga(dec, seed=7, max_evals=1234, crossover=cx, pop_size=50)
        assert r["n_evals"] <= 1234
        assert dec.n_evals == r["n_evals"]
        runs.append(r)
    a, b = runs
    assert a["best_cost"] == b["best_cost"]
    assert np.array_equal(a["best_perm"], b["best_perm"])
    assert a["trace_best"] == b["trace_best"]
    assert a["trace_evals"] == b["trace_evals"]
    assert a["generations"] == b["generations"]


def test_budget_smaller_than_population(p1):
    dec = Decoder(p1)
    r = run_ga(dec, seed=0, max_evals=10, pop_size=100)
    assert r["n_evals"] == 10
    assert sorted(r["best_perm"].tolist()) == list(range(1, p1.n + 1))


def test_budget_counts_only_this_run(p1):
    dec = Decoder(p1)
    dec.evaluate(np.arange(1, p1.n + 1))
    r = run_ga(dec, seed=0, max_evals=500, pop_size=50)
    assert r["n_evals"] == 500


def test_requires_a_budget_and_known_crossover(p1):
    dec = Decoder(p1)
    with pytest.raises(ValueError):
        run_ga(dec, seed=0)
    with pytest.raises(ValueError):
        run_ga(dec, seed=0, max_evals=100, crossover="CX")


@pytest.mark.parametrize("cx", ["OX", "PMX"])
def test_p1_feasible_and_cost_matches_validator(p1, cx):
    dec = Decoder(p1)
    t0 = time.perf_counter()
    r = run_ga(dec, seed=1, max_evals=20000, crossover=cx)
    dt = time.perf_counter() - t0
    print(f"\nP1 {cx}: best={r['best_cost']:.3f} evals={r['n_evals']} "
          f"gens={r['generations']} {r['n_evals'] / dt:.0f} evals/s")
    assert r["feasible"] is True
    routes = dec.routes(r["best_perm"])
    ok, msg = validate(p1, routes)
    assert ok, msg
    assert solution_cost(p1, routes) == pytest.approx(r["best_cost"])


def test_time_limit_and_monotone_trace(chao):
    run_ga(Decoder(chao), seed=0, max_evals=50)
    dec = Decoder(chao)
    t0 = time.perf_counter()
    r = run_ga(dec, seed=1, time_limit_s=1.0)
    dt = time.perf_counter() - t0
    assert 1.0 <= dt < 1.5
    tb = r["trace_best"]
    assert len(tb) >= 2
    assert len(tb) == len(r["trace_time"]) == len(r["trace_evals"])
    assert all(y <= x for x, y in zip(tb, tb[1:]))
    assert all(y >= x for x, y in zip(r["trace_time"], r["trace_time"][1:]))
    assert r["trace_evals"][-1] == r["n_evals"]
