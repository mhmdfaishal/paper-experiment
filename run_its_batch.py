"""Run the original ITS (R) on every instance x seed listed in config.json.

Each run writes ga_pso/results/raw/its/<instance>_s<seed>.json; finished runs are
skipped, so the script can be restarted safely.
Usage: python run_its_batch.py [--only P1.csv,P2.csv]
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ga_pso" / "results" / "raw" / "its"


def run_one(job):
    inst, seed, cfg = job
    stem = Path(inst["file"]).stem
    out = OUT / f"{stem}_s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    cmd = ["Rscript", "run_its.R", f"instances/{inst['file']}", str(inst["trucks"]),
           str(inst["trailers"]), str(cfg["its"]["iterations"]),
           str(cfg["its"]["max_time_s"]), str(seed), str(out)]
    p = subprocess.run(cmd, cwd=ROOT / "its_mcttrp", capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"FAIL {out.name}: {p.stderr[-300:]}")
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated instance files")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "ga_pso" / "config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    jobs = [(i, s, cfg) for i in cfg["instances"] for s in cfg["seeds"]
            if not only or i["file"] in only]
    # largest instances first so the slowest runs start early
    jobs.sort(key=lambda j: -int(j[0]["file"].startswith("CHAO")))
    # threads are enough: each job is a separate Rscript process
    with ThreadPoolExecutor(cfg["workers"]) as ex:
        for msg in ex.map(run_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
