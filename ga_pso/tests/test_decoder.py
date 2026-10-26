from pathlib import Path

import numpy as np
import pytest

from mcttrp.decoder import Decoder
from mcttrp.instance import load_instance
from mcttrp.solution import solution_cost, validate

INST = Path(__file__).resolve().parents[1] / "data"
CASES = [("P1.csv", 2, 1), ("P5.csv", 2, 1), ("P8.csv", 2, 1),
         ("CHAO_MCTTRP_01.txt", 8, 5), ("CHAO_MCTTRP_04.txt", 13, 8)]


@pytest.mark.parametrize("fname,nt,nl", CASES)
def test_feasible_decodes_validate_and_costs_match(fname, nt, nl):
    inst = load_instance(INST / fname, nt, nl)
    dec = Decoder(inst)
    rng = np.random.default_rng(0)
    n_feasible = 0
    for _ in range(200):
        perm = rng.permutation(np.arange(1, inst.n + 1))
        cost, feasible = dec.evaluate(perm)
        if not feasible:
            continue
        n_feasible += 1
        routes = dec.routes(perm)
        ok, msg = validate(inst, routes)
        assert ok, msg
        assert solution_cost(inst, routes) == pytest.approx(cost)
    assert n_feasible > 0


def test_infeasible_cost_is_worse_than_any_feasible():
    inst = load_instance(INST / "P1.csv", 2, 1)
    dec = Decoder(inst)
    rng = np.random.default_rng(1)
    feas, infeas = [], []
    for _ in range(300):
        c, f = dec.evaluate(rng.permutation(np.arange(1, inst.n + 1)))
        (feas if f else infeas).append(c)
    if feas and infeas:
        assert max(feas) < min(infeas)


def test_paper_optimum_order_decodes_to_207_or_better():
    # routes of the paper's Section 5.1 optimum, one after the other
    inst = load_instance(INST / "P8.csv", 2, 1)
    dec = Decoder(inst)
    cost, feasible = dec.evaluate(np.array([3, 2, 8, 7, 9, 1, 6, 5, 4, 10]))
    assert feasible
    assert cost <= 207 + 1e-9


def test_decoder_uses_same_fleet_rule_as_paper_heuristic():
    inst = load_instance(INST / "CHAO_MCTTRP_02.txt", 8, 5)
    dec = Decoder(inst)
    rng = np.random.default_rng(0)
    n_ptr_seen = []
    for _ in range(200):
        perm = rng.permutation(np.arange(1, inst.n + 1))
        _, feasible = dec.evaluate(perm)
        if feasible:
            routes = dec.routes(perm)
            n_ptr = sum(r.kind == "PTR" for r in routes)
            assert len(routes) <= 8 and len(routes) - n_ptr <= 5
            n_ptr_seen.append(n_ptr)
    assert max(n_ptr_seen) > 3


@pytest.mark.parametrize("fname,nt,nl", CASES)
def test_decoder_never_parks_trailer_at_depot(fname, nt, nl):
    inst = load_instance(INST / fname, nt, nl)
    dec = Decoder(inst)
    rng = np.random.default_rng(3)
    for _ in range(100):
        perm = rng.permutation(np.arange(1, inst.n + 1))
        for r in dec.routes(perm):
            assert all(root != 0 for root, _ in r.subtours)
