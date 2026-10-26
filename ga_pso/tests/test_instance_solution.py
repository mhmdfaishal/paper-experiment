import math
from pathlib import Path

import pytest

from mcttrp.instance import load_instance
from mcttrp.solution import Route, solution_cost, validate

INST = Path(__file__).resolve().parents[1] / "data"


def test_csv_instance_p1():
    inst = load_instance(INST / "P1.csv", n_trucks=2, n_trailers=1)
    assert inst.n == 8
    assert inst.n_vc == 4
    assert inst.dist[0, 1] == 20
    assert list(inst.demand[1]) == [3500, 0, 0, 2541]
    assert (inst.h_truck, inst.nh_truck) == (1500, 13)
    assert (inst.h_trailer, inst.nh_trailer) == (2000, 15)
    assert not inst.is_tc[4] and inst.is_tc[5]


def test_txt_instance_reorders_vc_first_like_r_code():
    inst = load_instance(INST / "CHAO_MCTTRP_01.txt", n_trucks=8, n_trailers=5)
    assert inst.n == 50
    # file customer 1 is a TC, so node 1 is file customer 2 at (49, 49)
    assert inst.dist[0, 1] == pytest.approx(math.sqrt(19 ** 2 + 9 ** 2))
    assert (inst.nh_truck, inst.nh_trailer) == (20, 10)
    assert all(not inst.is_tc[i] for i in range(1, inst.n_vc + 1))
    assert all(inst.is_tc[i] for i in range(inst.n_vc + 1, inst.n + 1))


def paper_p8_optimum():
    # optimal solution in Section 5.1 of the paper (207 km)
    return [
        Route("MVR", main=[3, 2, 1], subtours=[(2, [8, 7, 9])]),
        Route("PTR", main=[6, 5, 4, 10]),
    ]


def test_paper_example_cost_is_207():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    assert solution_cost(inst, paper_p8_optimum()) == pytest.approx(207)


def test_paper_example_is_valid():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    ok, msg = validate(inst, paper_p8_optimum())
    assert ok, msg


def test_missing_customer_is_invalid():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    sol = paper_p8_optimum()
    sol[1] = Route("PTR", main=[6, 5, 4])
    ok, msg = validate(inst, sol)
    assert not ok and "customer" in msg


def test_tc_on_main_tour_is_invalid():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    sol = [Route("MVR", main=[3, 2, 1, 8], subtours=[(2, [7, 9])]),
           Route("PTR", main=[6, 5, 4, 10])]
    ok, msg = validate(inst, sol)
    assert not ok and "TC" in msg


def test_fleet_limit_is_enforced():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    sol = paper_p8_optimum()[:1] + [Route("PTR", main=[6, 5]),
                                    Route("PTR", main=[4, 10])]
    ok, msg = validate(inst, sol)
    assert not ok and "fleet" in msg


def test_truck_hopper_capacity_is_enforced():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    ok, msg = validate(inst, [Route("PTR", main=list(range(1, 11)))])
    assert not ok and "hopper" in msg


def test_idle_trailer_frees_its_truck_for_a_ptr():
    # paper Table 4, instance 2 uses 4 PTRs with 8 trucks and 5 trailers
    inst = load_instance(INST / "P8.csv", n_trucks=3, n_trailers=2)
    sol = paper_p8_optimum()[:1] + [Route("PTR", main=[6, 5]),
                                    Route("PTR", main=[4, 10])]
    ok, msg = validate(inst, sol)
    assert ok, msg


def test_subtour_rooted_at_depot_is_invalid():
    inst = load_instance(INST / "P8.csv", n_trucks=2, n_trailers=1)
    sol = [Route("MVR", main=[3, 2, 1], subtours=[(0, [8, 7, 9])]),
           Route("PTR", main=[6, 5, 4, 10])]
    ok, msg = validate(inst, sol)
    assert not ok and "root" in msg
