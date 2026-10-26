import time
from pathlib import Path

import numpy as np
import pytest

from mcttrp.decoder import Decoder
from mcttrp.instance import load_instance
from mcttrp.solution import solution_cost, validate
from pso import inertia, keys_to_perm, run_pso, update_particle

INST = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def p1():
    inst = load_instance(INST / "P1.csv", 2, 1)
    return inst, Decoder(inst)


@pytest.fixture(scope="module")
def chao1():
    inst = load_instance(INST / "CHAO_MCTTRP_01.txt", 8, 5)
    return inst, Decoder(inst)


def test_keys_to_perm_is_valid_permutation():
    rng = np.random.default_rng(0)
    for n in (1, 5, 50):
        perm = keys_to_perm(rng.random(n))
        assert sorted(perm.tolist()) == list(range(1, n + 1))
    assert keys_to_perm(np.array([0.9, 0.1, 0.5])).tolist() == [2, 3, 1]


def test_update_clamps_velocity_and_keeps_positions_in_unit_cube():
    rng = np.random.default_rng(1)
    n, v_max = 30, 0.2
    x = rng.random(n)
    v = np.full(n, 50.0)
    pbest = rng.random(n)
    gbest = rng.random(n)
    for _ in range(50):
        x, v = update_particle(x, v, pbest, gbest, 0.9, 2.0, 2.0, v_max, rng)
        assert np.all(np.abs(v) <= v_max + 1e-12)
        assert np.all(x >= 0.0) and np.all(x <= 1.0)
    x, v = update_particle(x, np.full(n, -50.0), pbest, gbest, 0.9, 2.0, 2.0, v_max, rng)
    assert np.all(v >= -v_max - 1e-12)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)


def test_update_follows_pso_equation_without_clamping():
    rng = np.random.default_rng(2)
    x = np.array([0.5, 0.5])
    v = np.zeros(2)
    pbest = np.array([0.6, 0.4])
    gbest = np.array([0.6, 0.4])
    _, v2 = update_particle(x, v, pbest, gbest, 0.0, 2.0, 2.0, 10.0, rng)
    assert 0.0 <= v2[0] <= 0.4 and -0.4 <= v2[1] <= 0.0


def test_inertia_schedule_endpoints_and_monotone():
    assert inertia(0.0, 0.9, 0.4) == pytest.approx(0.9)
    assert inertia(1.0, 0.9, 0.4) == pytest.approx(0.4)
    assert inertia(0.5, 0.9, 0.4) == pytest.approx(0.65)
    assert inertia(1.7, 0.9, 0.4) == pytest.approx(0.4)


def test_requires_a_budget(p1):
    _, dec = p1
    with pytest.raises(ValueError):
        run_pso(dec, seed=0)


def test_max_evals_respected_exactly_and_deterministic(p1):
    _, dec = p1
    r1 = run_pso(dec, seed=3, max_evals=777, swarm_size=20)
    r2 = run_pso(dec, seed=3, max_evals=777, swarm_size=20)
    assert r1["n_evals"] == 777
    assert r1["n_evals"] <= 777
    assert r1["best_cost"] == r2["best_cost"]
    assert np.array_equal(r1["best_perm"], r2["best_perm"])
    assert r1["trace_best"] == r2["trace_best"]
    assert r1["trace_evals"] == r2["trace_evals"]
    assert r1["iterations"] == r2["iterations"]
    n0 = dec.n_evals
    r3 = run_pso(dec, seed=4, max_evals=100, swarm_size=20)
    assert dec.n_evals - n0 == r3["n_evals"] == 100


def test_budget_smaller_than_swarm(p1):
    _, dec = p1
    r = run_pso(dec, seed=0, max_evals=7, swarm_size=50)
    assert r["n_evals"] == 7
    assert sorted(r["best_perm"].tolist()) == list(range(1, dec.inst.n + 1))


def test_result_schema_and_trace_consistency(p1):
    _, dec = p1
    r = run_pso(dec, seed=0, max_evals=500, swarm_size=25)
    for k in ("best_cost", "best_perm", "feasible", "n_evals", "iterations",
              "trace_time", "trace_evals", "trace_best"):
        assert k in r
    assert len(r["trace_time"]) == len(r["trace_evals"]) == len(r["trace_best"])
    assert r["trace_evals"][-1] == r["n_evals"]
    assert r["trace_evals"] == sorted(r["trace_evals"])
    assert r["trace_time"] == sorted(r["trace_time"])
    assert r["iterations"] == 500 // 25


def test_p1_finds_valid_feasible_solution(p1):
    inst, dec = p1
    r = run_pso(dec, seed=1, max_evals=20000)
    print(f"\nP1 best cost @20000 evals: {r['best_cost']:.3f} (optimum 189)")
    assert r["feasible"] is True
    routes = dec.routes(r["best_perm"])
    ok, msg = validate(inst, routes)
    assert ok, msg
    assert solution_cost(inst, routes) == pytest.approx(r["best_cost"])


def test_time_limit_respected_and_trace_non_increasing(chao1):
    _, dec = chao1
    t0 = time.perf_counter()
    r = run_pso(dec, seed=1, time_limit_s=1.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.5
    tb = r["trace_best"]
    assert len(tb) >= 1
    assert all(b <= a for a, b in zip(tb, tb[1:]))
    assert r["trace_time"][-1] <= elapsed + 1e-6
    print(f"\nCHAO01 1s: {r['n_evals']} evals, best {r['best_cost']:.2f}, feasible={r['feasible']}")
