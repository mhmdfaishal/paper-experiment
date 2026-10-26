import json
from pathlib import Path

import pytest

import run_experiments as rx

ROOT = Path(__file__).resolve().parents[1]
P1 = {"file": "P1.csv", "trucks": 2, "trailers": 1, "ref": 189}


def test_budget_is_mean_its_time(tmp_path):
    for s, t in [(1, 4.0), (2, 6.0)]:
        (tmp_path / f"P1_s{s}.json").write_text(json.dumps({"time": t}))
    assert rx.its_budget(tmp_path, "P1") == pytest.approx(5.0)


def test_budget_missing_its_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rx.its_budget(tmp_path, "P1")


@pytest.mark.parametrize("alg", ["GA-OX", "GA-PMX", "PSO"])
def test_run_job_returns_validated_record(alg):
    rec = rx.run_job(alg, P1, seed=1, budget_s=0.5, params=rx.default_params(alg))
    assert rec["feasible"] and rec["valid"]
    assert rec["algorithm"] == alg and rec["instance"] == "P1"
    assert rec["time"] <= 0.5 + 0.3
    assert len(rec["trace_time"]) == len(rec["trace_best"]) > 0
    assert rec["cost"] >= 189 - 1e-9


def test_compact_trace_keeps_only_improvements():
    t, e, b = rx.compact_trace([0.1, 0.2, 0.3, 0.4, 0.5], [10, 20, 30, 40, 50],
                               [float("inf"), 9.0, 9.0, 7.0, 7.0])
    assert b == [float("inf"), 9.0, 7.0, 7.0]
    assert t == [0.1, 0.2, 0.4, 0.5] and e == [10, 20, 40, 50]
