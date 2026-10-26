import json
from pathlib import Path

import pytest

from mcttrp.instance import load_instance
from mcttrp.its_parse import parse_r_route, load_its_result
from mcttrp.solution import solution_cost, validate

HERE = Path(__file__).resolve().parent
INST = HERE.parent / "data"


def test_parse_cvr_with_vc_rooted_subtour():
    r = parse_r_route("CVR", [0, 4, 7, 6, 4, 2, 1, 3, 0])
    assert r.kind == "MVR" and r.main == [4, 2, 1, 3]
    assert r.subtours == [(4, [7, 6])]


def test_parse_pvr_and_ptr():
    assert parse_r_route("PVR", [0, 1, 2, 0]).main == [1, 2]
    assert parse_r_route("PTR", [0, 5, 8, 0]).kind == "PTR"


@pytest.mark.parametrize("fixture,fname,nt,nl", [
    ("its_P1.json", "P1.csv", 2, 1),
    ("its_C01.json", "CHAO_MCTTRP_01.txt", 8, 5),
])
def test_its_output_is_valid_and_cost_matches_r(fixture, fname, nt, nl):
    inst = load_instance(INST / fname, nt, nl)
    res = load_its_result(HERE / "fixtures" / fixture)
    ok, msg = validate(inst, res["routes"])
    assert ok, msg
    assert solution_cost(inst, res["routes"]) == pytest.approx(res["cost"])


def test_parse_two_subtours_from_same_parking_node():
    r = parse_r_route("CVR", [0, 10, 29, 5, 49, 61, 48, 5, 47, 64, 5, 0])
    assert r.main == [10, 29, 5]
    assert r.subtours == [(5, [49, 61, 48]), (5, [47, 64])]
