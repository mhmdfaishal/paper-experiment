"""Restore final result assets, optionally rerunning all configured experiments."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", help="recompute assets from raw results instead of using final presentation assets")
    parser.add_argument("--full", action="store_true", help="run ITS, GA/PSO and sensitivity before analysis")
    parser.add_argument("--fresh", action="store_true", help="archive existing raw results before a full run")
    args = parser.parse_args()
    if args.fresh and not args.full:
        parser.error("--fresh requires --full")
    raw = ROOT / "ga_pso/results/raw"
    if args.fresh and raw.exists():
        archive = raw.with_name("raw_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        raw.rename(archive)
        print("Previous results archived at", archive, flush=True)
    if args.full:
        run("run_its_batch.py")
        run("ga_pso/run_experiments.py")
        run("ga_pso/run_experiments.py", "--sensitivity")
    if args.full or args.analyze:
        run("ga_pso/compare.py")
    for name in (("tables", "figures") if args.full or args.analyze else ("figures",)):
        source = ROOT / "ga_pso/results" / ("" if args.full or args.analyze else "final_assets") / name
        target = ROOT / "ga_pso/results" / name
        if source != target:
            shutil.copytree(source, target, dirs_exist_ok=True)
    if not (args.full or args.analyze):
        for name in ("summary", "reproduction", "stats", "sensitivity"):
            shutil.copy2(ROOT / f"ga_pso/results/final_assets/{name}.csv",
                         ROOT / f"ga_pso/results/{name}.csv")
    run("render_results_readme.py")
    print("Final tables and figures are available in the root README.md")


if __name__ == "__main__":
    main()
