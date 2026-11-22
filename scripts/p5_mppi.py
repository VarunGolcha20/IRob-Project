"""Phase 5: MPPI swing-up with every model (PLAN Phase 5).

Stages
  tune   analytical model only, seeds 100-109; every config tried is appended to
         results/p5_tuning.csv; the first config with 10/10 success (in the order
         listed, plan default first) is frozen to results/p5_mppi_config.json.
  eval   frozen config, env seeds 0-19 (exact dm_control swingup initial states),
         models: analytical, thin-rod, MLP-B20, DeLaN-B20 (training seed with the
         best validation loss), stress test MLP-A20, DeLaN-A20.
         Per-episode results are cached in results/p5_runs/ (resumable).
  plots  figures/p5_success, p5_time_to_balance, p5_trajectories.

Plant: float64 analytical model + RK4 (= MuJoCo off the rail, see cartpole/mppi.py).

Usage:  python scripts/p5_mppi.py tune
        python scripts/p5_mppi.py eval [--workers 2] [--models ...]
        python scripts/p5_mppi.py plots
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from cartpole import mppi as mp  # noqa: E402
from cartpole import npdyn as nd  # noqa: E402

RUN_DIR = nd.RES_DIR / "p5_runs"
CFG_PATH = nd.RES_DIR / "p5_mppi_config.json"
TUNE_SEEDS = range(100, 110)
EVAL_SEEDS = range(0, 20)

TUNE_GRID = [
    dict(),                                  # PLAN 5.1 starting settings
    dict(K=256),
    dict(sigma=0.3),
    dict(sigma=0.8),
    dict(H=50),
]


def best_seed_name(kind, regime, N):
    info = json.loads((nd.RES_DIR / "p3_training.json").read_text())
    rows = [r for r in info["runs"] if r["kind"] == kind and r["regime"] == regime and r["N"] == N]
    return min(rows, key=lambda r: r["best_val"])["name"]


def get_model(name):
    p = nd.load_params()
    if name == "analytical":
        return nd.Analytical(p)
    if name == "thin_rod":
        return nd.Analytical(replace(p, I_c=p.m_p * (2 * p.l) ** 2 / 12.0))
    kind, cfg = name.split("-")
    return nd.load_model(best_seed_name(kind.lower(), cfg[0], int(cfg[1:])))


def _episode(args):
    model_name, seed, cfg_d = args
    path = RUN_DIR / f"{model_name}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = mp.MPPIConfig(**cfg_d)
    t0 = time.time()
    r = mp.run_episode(get_model(model_name), cfg, seed, record=True)
    r["balanced_curve"] = r["balanced_curve"].tolist()
    r["traj"] = {k: v.tolist() for k, v in r["traj"].items()}
    r["model"] = model_name
    r["wall_s"] = time.time() - t0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r))
    print(f"{model_name:<12} seed {seed:>3}  success {r['success']!s:<5} ttb {r['time_to_balance']}  "
          f"max|x| {r['max_abs_x']:.2f}  return {r['dmc_return']:.0f}  {r['wall_s']:.0f}s", flush=True)
    return r


def tune(_):
    A = get_model("analytical")
    path = nd.RES_DIR / "p5_tuning.csv"
    new = not path.exists()
    chosen = None
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["config", "n_success", "n", "median_ttb", "max_abs_x", "median_return", "ms_per_plan"])
        for delta in TUNE_GRID:
            cfg = replace(mp.MPPIConfig(lqr=False), **delta)
            rs = [mp.run_episode(A, cfg, s) for s in TUNE_SEEDS]
            ns = sum(r["success"] for r in rs)
            ttb = [r["time_to_balance"] for r in rs if r["time_to_balance"] is not None]
            row = [json.dumps(delta), ns, len(rs), float(np.median(ttb)) if ttb else None,
                   max(r["max_abs_x"] for r in rs), float(np.median([r["dmc_return"] for r in rs])),
                   float(np.mean([r["ms_per_plan"] for r in rs]))]
            w.writerow(row); fh.flush()
            print(row, flush=True)
            if chosen is None and ns == len(rs):
                chosen = cfg
                break                      # first config that passes is frozen; no further search
    if chosen is None:
        raise SystemExit("no config passed 10/10 on the tuning seeds")
    CFG_PATH.write_text(json.dumps(dict(config=chosen.as_dict(), tuned_on_seeds=list(TUNE_SEEDS)), indent=2))
    print("frozen:", CFG_PATH)


MODELS = ["analytical", "thin_rod", "DeLaN-B20", "MLP-B20", "DeLaN-A20", "MLP-A20"]


def evaluate(args):
    cfg_d = json.loads(CFG_PATH.read_text())["config"]
    cfg_d["lqr_in"], cfg_d["lqr_out"] = tuple(cfg_d["lqr_in"]), tuple(cfg_d["lqr_out"])
    models = args.models or MODELS
    jobs = [(m, s, cfg_d) for m in models for s in EVAL_SEEDS]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            list(pool.imap_unordered(_episode, jobs))
    else:
        for j in jobs:
            _episode(j)
    summarise()


def _frac_after(r):
    """Fraction of the time after the first balanced instant during which the pole is balanced."""
    b = np.asarray(r["balanced_curve"], bool)
    if not b.any():
        return 0.0
    return float(b[b.argmax():].mean())


def summarise():
    out = dict(config=json.loads(CFG_PATH.read_text())["config"], models={})
    for m in MODELS:
        rs = [json.loads((RUN_DIR / f"{m}_s{s}.json").read_text()) for s in EVAL_SEEDS
              if (RUN_DIR / f"{m}_s{s}.json").exists()]
        if not rs:
            continue
        k, n = sum(r["success"] for r in rs), len(rs)
        ttb = [r["time_to_balance"] for r in rs if r["success"]]
        out["models"][m] = dict(
            weights=None if m in ("analytical", "thin_rod") else best_seed_name(m.split("-")[0].lower(), m.split("-")[1][0], int(m.split("-")[1][1:])),
            n=n, n_success=k, success_rate=k / n, wilson95=mp.wilson(k, n),
            n_rail=sum(r["rail"] for r in rs),
            time_to_balance_median=float(np.median(ttb)) if ttb else None,
            time_to_balance_iqr=[float(v) for v in np.percentile(ttb, [25, 75])] if ttb else None,
            max_abs_x_max=float(max(r["max_abs_x"] for r in rs)),
            dmc_return_median=float(np.median([r["dmc_return"] for r in rs])),
            ms_per_plan_mean=float(np.mean([r["ms_per_plan"] for r in rs])),
            failures={r["seed"]: ("rail" if r["rail"] else "not balanced") for r in rs if not r["success"]},
            # stricter: success AND balanced over the whole final 2 s (catches balance-then-fall, ISSUES E5)
            n_held_to_end=sum(r["success"] and r["frac_balanced_last2s"] == 1.0 for r in rs),
            wilson95_held=mp.wilson(sum(r["success"] and r["frac_balanced_last2s"] == 1.0 for r in rs), n),
            frac_time_balanced_after_first_median=float(np.median([_frac_after(r) for r in rs])))
    (nd.RES_DIR / "p5_closed_loop.json").write_text(json.dumps(out, indent=2))
    for m, v in out["models"].items():
        print(f"{m:<12} {v['n_success']}/{v['n']}  Wilson {v['wilson95'][0]:.2f}-{v['wilson95'][1]:.2f}  "
              f"ttb {v['time_to_balance_median']}  rail {v['n_rail']}  max|x| {v['max_abs_x_max']:.2f}  "
              f"return {v['dmc_return_median']:.0f}  {v['ms_per_plan_mean']:.1f} ms/plan  "
              f"held-to-end {v['n_held_to_end']}/{v['n']}  frac-bal-after {v['frac_time_balanced_after_first_median']:.3f}")


def plots(_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.loads((nd.RES_DIR / "p5_closed_loop.json").read_text())["models"]
    C = {"analytical": "#1b9e77", "thin_rod": "#66a61e", "DeLaN-B20": "#7570b3", "MLP-B20": "#d95f02",
         "DeLaN-A20": "#b2abd2", "MLP-A20": "#fdb863"}
    names = [m for m in MODELS if m in res]
    fig_dir = nd.ROOT / "figures"

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(names):
        v = res[m]
        lo, hi = v["wilson95"]
        ax.bar(i, v["success_rate"], color=C[m], alpha=0.85)
        ax.errorbar(i, v["success_rate"], yerr=[[v["success_rate"] - lo], [hi - v["success_rate"]]], color="k", capsize=5)
        ax.text(i, 0.02, f"{v['n_success']}/{v['n']}", ha="center", color="w", fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set(ylim=(0, 1.05), ylabel="success rate (Wilson 95 % CI)",
           title="MPPI swing-up: success over env seeds 0-19 (balanced for 2 s, rail never touched)")
    save(fig, "p5_success")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    t = np.linspace(0, 10, 501)
    for m in names:
        rs = [json.loads((RUN_DIR / f"{m}_s{s}.json").read_text()) for s in EVAL_SEEDS]
        ttb = np.array([r["time_to_balance"] if r["success"] else np.inf for r in rs])
        axes[0].step(t, [(ttb <= tt).mean() for tt in t], where="post", color=C[m], lw=2, label=m)
        B = np.zeros((len(rs), 1000))
        for i, r in enumerate(rs):
            b = np.asarray(r["balanced_curve"], float)
            B[i, :len(b)] = b
        axes[1].plot(np.arange(1000) * 0.01, B.mean(0), color=C[m], lw=1.8, label=m)
    axes[0].set(xlabel="time [s]", ylabel="fraction of seeds",
                title="Time to balance: first time balanced and held 2 s\n(failures censored at 10 s)", xlim=(0, 10), ylim=(0, 1.02))
    axes[1].set(xlabel="time [s]", ylabel="fraction of seeds balanced at t",
                title="Instantaneous balance: $|\\theta| < 0.2$ and $|\\dot\\theta| < 1$\n(drops after first balance = pole fell again)", xlim=(0, 10), ylim=(0, 1.02))
    axes[0].legend()
    save(fig, "p5_time_to_balance")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    traj_seed = 0
    for m in names:
        r = json.loads((RUN_DIR / f"{m}_s{traj_seed}.json").read_text())["traj"]
        axes[0].plot(r["t"], r["th"], color=C[m], lw=1.5, label=m)
        axes[1].plot(r["t"], r["x"], color=C[m], lw=1.5)
        axes[2].plot(r["t"], r["u"], color=C[m], lw=0.8)
    axes[0].set_ylabel(r"wrapped $\theta$ [rad]"); axes[1].set_ylabel("x [m]"); axes[2].set_ylabel("u")
    axes[1].axhline(1.8, color="r", ls=":", lw=1); axes[1].axhline(-1.8, color="r", ls=":", lw=1)
    axes[2].set_xlabel("time [s]"); axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title("Closed-loop trajectories, env seed 0")
    save(fig, "p5_trajectories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "eval", "plots", "summary"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    {"tune": tune, "eval": evaluate, "plots": plots, "summary": lambda a: summarise()}[args.stage](args)


if __name__ == "__main__":
    main()
