"""Run GA and PSO with the same time budget as ITS on each instance."""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(HERE))

from ga import run_ga                       # noqa: E402
from pso import run_pso                     # noqa: E402
from mcttrp.decoder import Decoder          # noqa: E402
from mcttrp.instance import load_instance   # noqa: E402
from mcttrp.solution import solution_cost, validate  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())
RAW = ROOT / "results" / "raw"
ALGS = ["GA-OX", "GA-PMX", "PSO"]


def its_budget(its_dir, stem):
    """Mean ITS run time on one instance."""
    times = [json.loads(p.read_text())["time"] for p in Path(its_dir).glob(f"{stem}_s*.json")]
    if not times:
        raise FileNotFoundError(f"no ITS results for {stem} in {its_dir}")
    return float(np.mean(times))


def compact_trace(times, evals, best):
    """Keep only the points where the best cost changes, plus the last one."""
    keep = [i for i in range(len(best)) if i in (0, len(best) - 1) or best[i] != best[i - 1]]
    return [times[i] for i in keep], [evals[i] for i in keep], [best[i] for i in keep]


def default_params(alg):
    if alg.startswith("GA"):
        p = {k: v for k, v in CFG["ga"].items() if k != "variants"}
        p["crossover"] = alg.split("-")[1]
        return p
    return dict(CFG["pso"])


def run_job(alg, inst_cfg, seed, budget_s, params):
    inst = load_instance(ROOT / CFG["instances_dir"] / inst_cfg["file"],
                         inst_cfg["trucks"], inst_cfg["trailers"])
    dec = Decoder(inst)
    # warm-up so numba compilation is not counted in the budget
    run = run_ga if alg.startswith("GA") else run_pso
    run(dec, seed, max_evals=50, **params)
    dec.n_evals = 0
    t0 = time.perf_counter()
    res = run(dec, seed, time_limit_s=budget_s, **params)
    elapsed = time.perf_counter() - t0
    routes = dec.routes(res["best_perm"])
    tt, te, tb = compact_trace(res["trace_time"], res["trace_evals"], res["trace_best"])
    ok, msg = validate(inst, routes)
    cost = solution_cost(inst, routes)
    return {
        "algorithm": alg, "instance": inst.name, "seed": seed, "budget_s": budget_s,
        "params": params, "cost": cost, "feasible": bool(res["feasible"]), "valid": ok,
        "valid_msg": msg, "time": elapsed, "n_evals": int(res["n_evals"]),
        "routes": [{"kind": r.kind, "main": r.main, "subtours": r.subtours} for r in routes],
        "trace_time": [float(x) for x in tt],
        "trace_evals": [int(x) for x in te],
        "trace_best": [float(x) for x in tb],
    }


def _worker(job):
    alg, inst_cfg, seed, budget, params, out = job
    if out.exists():
        return f"skip {out.name}"
    rec = run_job(alg, inst_cfg, seed, budget, params)
    out.write_text(json.dumps(rec))
    flag = "" if rec["valid"] else f"  INVALID: {rec['valid_msg']}"
    return f"{alg} {rec['instance']} s{seed} cost={rec['cost']:.2f} evals={rec['n_evals']}{flag}"


def main_jobs(only=()):
    jobs = []
    for inst in CFG["instances"]:
        if only and inst["file"] not in only:
            continue
        stem = Path(inst["file"]).stem
        budget = its_budget(RAW / "its", stem)
        for alg in ALGS:
            (RAW / alg).mkdir(parents=True, exist_ok=True)
            for s in CFG["seeds"]:
                jobs.append((alg, inst, s, budget, default_params(alg),
                             RAW / alg / f"{stem}_s{s}.json"))
    return jobs


def sensitivity_jobs():
    sc = CFG["sensitivity"]
    inst = next(i for i in CFG["instances"] if i["file"] == sc["instance"])
    stem = Path(inst["file"]).stem
    budget = its_budget(RAW / "its", stem)
    out_dir = RAW / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in sc["ga_mutation_rate"]:
        for s in sc["seeds"]:
            p = default_params("GA-OX") | {"mutation_rate": m}
            jobs.append(("GA-OX", inst, s, budget, p, out_dir / f"GA-OX_mut{m}_{stem}_s{s}.json"))
    for w in sc["pso_w_end"]:
        for s in sc["seeds"]:
            p = default_params("PSO") | {"w_end": w}
            jobs.append(("PSO", inst, s, budget, p, out_dir / f"PSO_wend{w}_{stem}_s{s}.json"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated instance files")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    jobs = sensitivity_jobs() if args.sensitivity else main_jobs(only)
    jobs.sort(key=lambda j: -j[3])
    with ProcessPoolExecutor(CFG["workers"]) as ex:
        for msg in ex.map(_worker, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
