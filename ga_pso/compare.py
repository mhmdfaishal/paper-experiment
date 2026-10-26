"""Tables, statistical tests and figures for ITS, GA-OX, GA-PMX and PSO."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from scipy.stats import ranksums  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(HERE))
from mcttrp.instance import load_instance              # noqa: E402
from mcttrp.its_parse import load_its_result           # noqa: E402
from mcttrp.solution import Route, solution_cost, validate  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
ALGS = ["ITS", "GA-OX", "GA-PMX", "PSO"]

PAPER = {
    "P1": (None, 189, 5.03), "P2": (None, 142, 3.88), "P3": (None, 106, 3.90),
    "P4": (None, 256, 5.58), "P5": (None, 109, 4.72), "P6": (None, 223, 6.48),
    "P7": (None, 238, 7.97), "P8": (None, 213, 6.09),
    "CHAO_MCTTRP_01": (662.93, 640.66, 5.58 * 60), "CHAO_MCTTRP_02": (832.60, 708.08, 5.28 * 60),
    "CHAO_MCTTRP_03": (885.79, 764.81, 5.76 * 60), "CHAO_MCTTRP_04": (1013.57, 908.53, 10.55 * 60),
    "CHAO_MCTTRP_05": (1168.01, 1012.83, 8.93 * 60), "CHAO_MCTTRP_06": (1264.09, 1101.90, 14.13 * 60),
}

STYLE = {
    "ITS":    dict(color="#2a78d6", marker="o", ls="-"),
    "GA-OX":  dict(color="#eb6834", marker="s", ls="--"),
    "GA-PMX": dict(color="#1baf7a", marker="^", ls="-."),
    "PSO":    dict(color="#eda100", marker="D", ls=":"),
}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5})


def short(stem):
    return stem.replace("CHAO_MCTTRP_", "C")


def inst_cfg(stem):
    return next(i for i in CFG["instances"] if Path(i["file"]).stem == stem)


def load_inst(stem):
    c = inst_cfg(stem)
    return load_instance(ROOT / CFG["instances_dir"] / c["file"], c["trucks"], c["trailers"])


def load_runs():
    rows, cache = [], {}
    for path in sorted((RAW / "its").glob("*.json")):
        res = load_its_result(path)
        stem = Path(res["instance"]).stem
        inst = cache.setdefault(stem, load_inst(stem))
        ok, msg = validate(inst, res["routes"])
        rows.append(dict(algorithm="ITS", instance=stem, seed=res["seed"],
                         cost=solution_cost(inst, res["routes"]), valid=ok, time=res["time"],
                         n_evals=np.nan, init_cost=res["init_cost"], routes=res["routes"],
                         trace_time=res["trace"]["time"], trace_best=res["trace"]["best"]))
    for alg in ALGS[1:]:
        for path in sorted((RAW / alg).glob("*.json")):
            d = json.loads(path.read_text())
            inst = cache.setdefault(d["instance"], load_inst(d["instance"]))
            routes = [Route(r["kind"], r["main"], [tuple(s) for s in r["subtours"]])
                      for r in d["routes"]]
            ok, _ = validate(inst, routes)
            rows.append(dict(algorithm=alg, instance=d["instance"], seed=d["seed"],
                             cost=solution_cost(inst, routes), valid=ok and d["feasible"],
                             time=d["time"], n_evals=d["n_evals"], init_cost=np.nan,
                             routes=routes, trace_time=d["trace_time"], trace_best=d["trace_best"]))
    return pd.DataFrame(rows)


def summary_table(df):
    out = []
    for (stem, alg), g in df.groupby(["instance", "algorithm"]):
        ref = inst_cfg(stem)["ref"]
        c = g.loc[g.valid, "cost"]
        gap = lambda x: 100.0 * (x - ref) / ref   # noqa: E731
        out.append(dict(instance=stem, algorithm=alg, runs=len(g), valid_runs=int(g.valid.sum()),
                        ref=ref, best=c.min(), mean=c.mean(), std=c.std(ddof=1), worst=c.max(),
                        gap_best=gap(c.min()), gap_mean=gap(c.mean()), gap_worst=gap(c.max()),
                        time_mean=g.time.mean(), evals_mean=g.n_evals.mean()))
    s = pd.DataFrame(out)
    order = {Path(i["file"]).stem: k for k, i in enumerate(CFG["instances"])}
    s["_o"] = s.instance.map(order) * 10 + s.algorithm.map({a: k for k, a in enumerate(ALGS)})
    return s.sort_values("_o").drop(columns="_o")


def reproduction_table(df):
    out = []
    for stem, g in df[df.algorithm == "ITS"].groupby("instance"):
        cw, its, t = PAPER[stem]
        out.append(dict(instance=stem, paper_cw=cw, ours_cw=g.init_cost.iloc[0],
                        paper_its=its, ours_best=g.cost.min(), ours_mean=g.cost.mean(),
                        diff_best_pct=100 * (g.cost.min() - its) / its,
                        paper_time_s=t, ours_time_s=g.time.mean()))
    r = pd.DataFrame(out)
    order = {Path(i["file"]).stem: k for k, i in enumerate(CFG["instances"])}
    return r.sort_values("instance", key=lambda s: s.map(order))


def stats_table(df, summ):
    """Wilcoxon rank-sum tests: ITS vs best other, OX vs PMX, best GA vs PSO."""
    out = []
    for stem, g in df[df.valid].groupby("instance"):
        cost = {a: g.loc[g.algorithm == a, "cost"].values for a in ALGS}
        means = summ[summ.instance == stem].set_index("algorithm")["mean"]
        best_meta = means.drop("ITS").idxmin()
        best_ga = means[["GA-OX", "GA-PMX"]].idxmin()
        for a, b in [("ITS", best_meta), ("GA-OX", "GA-PMX"), (best_ga, "PSO")]:
            x, y = cost[a], cost[b]
            if np.allclose(np.r_[x, y], x[0]):
                p = 1.0
            else:
                p = ranksums(x, y).pvalue
            out.append(dict(instance=stem, a=a, b=b, mean_a=x.mean(), mean_b=y.mean(),
                            p_value=p, significant_0_05=p < 0.05))
    return pd.DataFrame(out)


def sensitivity_table():
    rows = []
    for path in sorted((RAW / "sensitivity").glob("*.json")):
        d = json.loads(path.read_text())
        param, val = (("mutation_rate", d["params"]["mutation_rate"]) if d["algorithm"] == "GA-OX"
                      else ("w_end", d["params"]["w_end"]))
        rows.append(dict(algorithm=d["algorithm"], param=param, value=val, seed=d["seed"],
                         cost=d["cost"], valid=d["valid"] and d["feasible"]))
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)
    ref = inst_cfg(Path(CFG["sensitivity"]["instance"]).stem)["ref"]
    agg = s.groupby(["algorithm", "param", "value"]).cost.agg(["mean", "std", "min", "count"]).reset_index()
    agg["gap_mean"] = 100 * (agg["mean"] - ref) / ref
    return agg


def _step_mean(runs, grid):
    """Average best-so-far curve of several runs on one time grid."""
    vals = []
    for t, b in runs:
        t, b = np.asarray(t, float), np.asarray(b, float)
        b = np.where(np.isfinite(b), b, np.nan)
        idx = np.searchsorted(t, grid, side="right") - 1
        v = np.where(idx >= 0, b[np.clip(idx, 0, None)], np.nan)
        vals.append(v)
    with np.errstate(all="ignore"):
        return np.nanmean(np.array(vals), axis=0)


def fig_convergence(df, stems, name):
    fig, axes = plt.subplots(1, len(stems), figsize=(7.16, 2.2))
    for ax, stem in zip(np.atleast_1d(axes), stems):
        g = df[df.instance == stem]
        tmax = g.time.max()
        grid = np.linspace(0, tmax, 300)
        for alg in ALGS:
            runs = [(r.trace_time, r.trace_best) for r in g[g.algorithm == alg].itertuples()]
            if not runs:
                continue
            m = _step_mean(runs, grid)
            st = STYLE[alg]
            ax.plot(grid, m, color=st["color"], ls=st["ls"], label=alg,
                    marker=st["marker"], markevery=50, markersize=4)
        ax.axhline(inst_cfg(stem)["ref"], color="#52514e", lw=0.8, ls=(0, (1, 2)))
        ax.set_title(short(stem))
        ax.set_xlabel("Waktu (s)")
    np.atleast_1d(axes)[0].set_ylabel("Rata-rata biaya layak terbaik")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, name)


def fig_gap(summ):
    stems = [Path(i["file"]).stem for i in CFG["instances"]]
    x = np.arange(len(stems))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.16, 2.3))
    for k, alg in enumerate(ALGS):
        s = summ[summ.algorithm == alg].set_index("instance").reindex(stems)
        ax.bar(x + (k - 1.5) * w, s.gap_mean, w * 0.9, color=STYLE[alg]["color"], label=alg,
               yerr=100 * s["std"] / s["ref"], error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(0, color="#0b0b0b", lw=0.6)
    ax.set_xticks(x, [short(s) for s in stems])
    ax.set_ylabel("Rata-rata gap terhadap acuan (%)")
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    _save(fig, "gap_by_instance")


def fig_sensitivity(sens):
    if sens.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
    for ax, (alg, param, label) in zip(axes, [("GA-OX", "mutation_rate", "Peluang mutasi GA-OX"),
                                               ("PSO", "w_end", "Inersia akhir PSO $w_{end}$")]):
        s = sens[(sens.algorithm == alg)].sort_values("value")
        st = STYLE[alg]
        ax.errorbar(s.value, s.gap_mean, yerr=100 * s["std"] / inst_cfg(
            Path(CFG["sensitivity"]["instance"]).stem)["ref"], color=st["color"],
            marker=st["marker"], ms=4, capsize=2, lw=1.2)
        if param == "mutation_rate":
            ax.set_xscale("log", base=2)
            ax.minorticks_off()
        ax.set_xticks(s.value, [f"{v:g}".replace(".", ",") for v in s.value])
        ax.set_xlabel(label)
    axes[0].set_ylabel("Rata-rata gap (%)")
    fig.tight_layout()
    _save(fig, "sensitivity")


def fig_routes(df, stem):
    """Plot the best ITS solution next to the best GA/PSO solution."""
    inst = load_inst(stem)
    path = ROOT / CFG["instances_dir"] / inst_cfg(stem)["file"]
    data = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    xy = {0: (float(data[2][1]), float(data[2][2]))}
    types = [0] + [int(r[3]) for r in data[3:3 + inst.n]]
    order = [0] + [i for i in range(1, inst.n + 1) if types[i] == 0] + \
            [i for i in range(1, inst.n + 1) if types[i] == 1]
    for new, old in enumerate(order[1:], start=1):
        xy[new] = (float(data[2 + old][1]), float(data[2 + old][2]))
    g = df[(df.instance == stem) & df.valid]
    meta = g[g.algorithm != "ITS"].sort_values("cost").iloc[0]
    its = g[g.algorithm == "ITS"].sort_values("cost").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for ax, row in zip(axes, [its, meta]):
        for r in row.routes:
            col = {"PTR": "#6250d6", "PVR": "#2a78d6", "MVR": "#2a78d6"}[r.kind]
            tour = [0] + r.main + [0]
            ax.plot(*zip(*[xy[v] for v in tour]), color=col, lw=1.0)
            for root, nodes in r.subtours:
                st = [root] + nodes + [root]
                ax.plot(*zip(*[xy[v] for v in st]), color="#eb6834", lw=1.0, ls="--")
        vc = [xy[i] for i in range(1, inst.n + 1) if not inst.is_tc[i]]
        tc = [xy[i] for i in range(1, inst.n + 1) if inst.is_tc[i]]
        ax.scatter(*zip(*vc), s=10, color="#0b0b0b", zorder=3, label="VC")
        ax.scatter(*zip(*tc), s=12, facecolor="white", edgecolor="#0b0b0b", zorder=3, label="TC")
        ax.scatter(*xy[0], s=40, marker="s", color="#0b0b0b", zorder=4, label="Depot")
        ax.set_title(f"{row.algorithm} (biaya {row.cost:.2f})".replace(".", ","))
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#2a78d6", label="Main tour kendaraan lengkap"),
               Line2D([], [], color="#eb6834", ls="--", label="Sub-tour truk"),
               Line2D([], [], color="#6250d6", label="Rute truk murni (PTR)")]
    fig.legend(handles=handles + axes[0].get_legend_handles_labels()[0],
               loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, f"routes_{short(stem)}")


TAB = ROOT / "results" / "tables"


def _f(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{d}f}".replace(".", ",")


def _tex_name(stem):
    return short(stem).replace("_", r"\_")


def write_tex_tables(summ, repro, stats, sens):
    TAB.mkdir(parents=True, exist_ok=True)
    lines = []
    for stem in [Path(i["file"]).stem for i in CFG["instances"]]:
        s = summ[summ.instance == stem]
        if s.empty:
            continue
        best_mean = s["mean"].min()
        for k, r in enumerate(s.itertuples()):
            mean = _f(r.mean)
            if abs(r.mean - best_mean) < 1e-9:
                mean = r"\textbf{" + mean + "}"
            head = f"{_tex_name(stem)} & {_f(r.ref)}" if k == 0 else " & "
            lines.append(f"{head} & {r.algorithm} & {_f(r.best)} & {mean} & {_f(r.std)} & "
                         f"{_f(r.worst)} & {_f(r.gap_best)} & {_f(r.gap_mean)} & {_f(r.time_mean, 1)} \\\\")
        lines.append(r"\midrule")
    (TAB / "summary.tex").write_text("\n".join(lines[:-1]) + "\n")
    # The report separates real cases and Chao cases into two tables.
    blocks = "\n".join(lines[:-1]).split("\\midrule\n")
    for name, prefix in [("summary-real.tex", "P"), ("summary-chao.tex", "C")]:
        selected = [block for block in blocks if block.startswith(prefix)]
        (TAB / name).write_text("\\midrule\n".join(selected) + "\n")

    lines = []
    for r in repro.itertuples():
        lines.append(f"{_tex_name(r.instance)} & {_f(r.paper_cw)} & {_f(r.ours_cw)} & {_f(r.paper_its)} & "
                     f"{_f(r.ours_best)} & {_f(r.ours_mean)} & {_f(r.diff_best_pct)} & "
                     f"{_f(r.paper_time_s, 1)} & {_f(r.ours_time_s, 1)} \\\\")
    (TAB / "reproduction.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for stem in [Path(i["file"]).stem for i in CFG["instances"]]:
        t = stats[stats.instance == stem] if not stats.empty else stats
        if t.empty:
            continue
        cells = []
        for k, (_, r) in enumerate(t.iterrows()):
            p = r.p_value
            ptxt = "$<$0,001" if p < 1e-3 else f"{p:.3f}".replace(".", ",")
            if p < 0.05:
                ptxt = r"\textbf{" + ptxt + "}"
            name = r.b if k == 0 else (r.a if k == 2 else "")
            cells.append(f"{ptxt} ({name})" if name else ptxt)
        lines.append(f"{_tex_name(stem)} & " + " & ".join(cells) + r" \\")
    (TAB / "stats.tex").write_text("\n".join(lines) + "\n")
    if not sens.empty:
        lines = []
        for r in sens.sort_values(["algorithm", "value"]).itertuples():
            label = {"mutation_rate": "peluang mutasi", "w_end": r"$w_{\text{end}}$"}[r.param]
            lines.append(f"{r.algorithm} & {label} = {str(r.value).replace('.', ',')} & {_f(r.mean)} & "
                         f"{_f(r.std)} & {_f(r.min)} & {_f(r.gap_mean)} \\\\")
        (TAB / "sensitivity.tex").write_text("\n".join(lines) + "\n")
        compact = []
        for r in sens.sort_values(["algorithm", "value"]).itertuples():
            compact.append(f"{r.algorithm} & {str(r.value).replace('.', ',')} & {_f(r.mean)} & "
                           f"{_f(r.std)} & {_f(r.min)} & {_f(r.gap_mean)} " + r"\\")
        (TAB / "sensitivity-compact.tex").write_text("\n".join(compact) + "\n")



def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_runs()
    out = ROOT / "results"
    raw = df.drop(columns=["routes", "trace_time", "trace_best"])
    raw.to_csv(out / "all_runs.csv", index=False)
    summ = summary_table(df)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    repro = reproduction_table(df)
    repro.to_csv(out / "reproduction.csv", index=False, float_format="%.4f")
    full = df.groupby("instance").algorithm.nunique()
    complete = full[full == len(ALGS)].index
    stats = stats_table(df[df.instance.isin(complete)], summ)
    stats.to_csv(out / "stats.csv", index=False, float_format="%.6f")
    sens = sensitivity_table()
    if not sens.empty:
        sens.to_csv(out / "sensitivity.csv", index=False, float_format="%.4f")
    if len(complete):
        conv = [s for s in ["CHAO_MCTTRP_01", "CHAO_MCTTRP_04", "CHAO_MCTTRP_06"] if s in complete]
        fig_convergence(df[df.instance.isin(complete)], conv, "convergence")
        fig_gap(summ[summ.instance.isin(complete)])
        if "CHAO_MCTTRP_01" in complete:
            fig_routes(df, "CHAO_MCTTRP_01")
    fig_sensitivity(sens)
    write_tex_tables(summ, repro, stats, sens)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    n_invalid = int((~df.valid).sum())
    print(f"\nruns: {len(df)}  invalid: {n_invalid}")


if __name__ == "__main__":
    main()
