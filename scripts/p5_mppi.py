"""Phase 5: closed-loop MPPI swing-up with every model (PLAN 5.3).

The SAME frozen MPPI configuration (results/p5_mppi_config.json, from scripts/p5_tune.py) drives
the analytical model, the thin-rod model, and the learned models, on env seeds 0-19 (the seed sets
the initial state and the MPPI noise).

Models:
    analytical, thin_rod, mlp_B20, delan_B20      (learned: the seed with the best VALIDATION loss)
    --stress   adds mlp_A20, delan_A20            (never saw the upright region)
    --all-seeds uses all three training seeds of each learned model instead of the best one
Success = balanced (|wrap th| < 0.2, |thd| < 1) for 2 s within the 10 s episode AND the rail limit never active.

Stages:
    run     simulate every (model, seed); resumable (a finished run's .npz is never repeated)
    report  Wilson intervals, time-to-balance, figures, results/p5_closed_loop.json
Usage:  python scripts/p5_mppi.py [--stage all|run|report] [--workers 4] [--seeds 0-19] [--stress] [--all-seeds]
Acceptance: analytical model >= 18/20 successes; otherwise the problem is the controller, not the models.
Outputs: results/p5_runs/*.npz, results/p5_closed_loop.json, figures/p5_success.*,
         figures/p5_time_to_balance.*, figures/p5_trajectories.*
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cartpole.closedloop import episode_job, wilson  # noqa: E402
from cartpole.mppi import MPPIConfig  # noqa: E402
from cartpole.plotting import MODEL_COLORS, RES_DIR, savefig, setup_style  # noqa: E402

RUN_DIR = RES_DIR / "p5_runs"
EPISODE_S = 10.0


def parse_seeds(s: str):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def load_config(use_default: bool) -> dict:
    path = RES_DIR / "p5_mppi_config.json"
    if use_default or not path.exists():
        if not use_default:
            print("WARNING: results/p5_mppi_config.json not found (run scripts/p5_tune.py first); using defaults")
        return MPPIConfig().to_dict()
    return json.loads(path.read_text())["config"]


def model_specs(stress: bool, all_seeds: bool):
    from cartpole.loading import best_run, grid_runs
    specs = [("analytical", "analytical"), ("thin_rod", "thin_rod")]
    regimes = ["B"] + (["A"] if stress else [])
    for regime in regimes:
        for kind in ("mlp", "delan"):
            if all_seeds:
                for r in grid_runs():
                    if r["kind"] == kind and r["regime"] == regime and r["N"] == 20:
                        specs.append((r["name"], r["path"]))
            else:
                specs.append((f"{kind}_{regime}20", best_run(kind, regime, 20)["path"]))
    return specs


def stage_run(args):
    cfg = load_config(args.default_config)
    seeds = parse_seeds(args.seeds)
    specs = model_specs(args.stress, args.all_seeds)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # slowest models first so the pool finishes evenly
    order = {"delan": 0, "mlp": 1, "thin_rod": 2, "analytical": 3}
    specs.sort(key=lambda s: order.get(s[0].split("_")[0], 4))
    jobs = [dict(spec=spec, cfg=cfg, seed=s, name=name, save_dir=str(RUN_DIR), seconds=EPISODE_S)
            for name, spec in specs for s in seeds]
    print(f"{len(jobs)} episodes ({len(specs)} models x {len(seeds)} seeds), {args.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(episode_job, jobs, chunksize=1), 1):
            print(f"[{i}/{len(jobs)}] {r['name']:<16} seed {r['seed']:>2}: success {r['success']!s:<5} "
                  f"ttb {r['time_to_balance']} max|x| {r['max_abs_x']:.2f} limit {r['limit_hit']} "
                  f"{r['ms_per_step']:.0f} ms/step  ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------
def collect(names=None):
    runs = {}
    for f in sorted(RUN_DIR.glob("*_seed*.npz")):
        name, seed = f.stem.rsplit("_seed", 1)
        if names is not None and name not in names:
            continue
        z = np.load(f)
        runs.setdefault(name, {})[int(seed)] = dict(score=json.loads(str(z["score"])), path=f)
    return runs


def style(name):
    if name in ("analytical", "thin_rod"):
        return dict(color=MODEL_COLORS[name], ls="-" if name == "analytical" else "-.")
    kind, regime = name.split("_")[0], name.split("_")[1][0]
    return dict(color=MODEL_COLORS[kind], ls="--" if regime == "A" else "-")


def label(name):
    if name == "analytical":
        return "Analytical"
    if name == "thin_rod":
        return "Analytical (thin rod)"
    parts = name.split("_")
    kind = "MLP" if parts[0] == "mlp" else "DeLaN"
    return f"{kind} {parts[1]}" + (f" (s{parts[2][1:]})" if len(parts) > 2 and parts[2].startswith("s") else "")


def summarise(runs):
    out = {}
    for name, rs in runs.items():
        sc = [r["score"] for r in rs.values()]
        n, k = len(sc), sum(s["success"] for s in sc)
        lo, hi = wilson(k, n)
        ttb = [s["time_to_balance"] for s in sc if s["success"]]
        out[name] = dict(
            n_runs=n, n_success=k, success_rate=k / n, wilson95=[lo, hi],
            mean_ttb_s=float(np.mean(ttb)) if ttb else None, median_ttb_s=float(np.median(ttb)) if ttb else None,
            balanced_but_rail_hit=int(sum(s["time_to_balance"] is not None and s["limit_hit"] for s in sc)),
            rail_hits=int(sum(s["limit_hit"] for s in sc)),
            mean_max_abs_x=float(np.mean([s["max_abs_x"] for s in sc])),
            mean_return=float(np.mean([s["ret"] for s in sc])),
            mean_ms_per_step=float(np.mean([s["ms_per_step"] for s in sc])),
            mean_lqr_switches=float(np.mean([s["lqr_switches"] for s in sc])),
            seeds=sorted(rs))
    return out


def order_names(names):
    pref = ["analytical", "thin_rod"]
    rest = sorted(n for n in names if n not in pref)
    return [n for n in pref if n in names] + rest


def stage_report(args):
    setup_style()
    import matplotlib.pyplot as plt
    runs = collect()
    if not runs:
        sys.exit("no runs found in results/p5_runs (run --stage run first)")
    names = order_names(runs)
    summ = summarise(runs)
    for n in names:
        s = summ[n]
        print(f"{n:<16} {s['n_success']:>2}/{s['n_runs']:<2} success  Wilson [{s['wilson95'][0]:.2f}, {s['wilson95'][1]:.2f}]  "
              f"mean ttb {s['mean_ttb_s']}  rail hits {s['rail_hits']}  {s['mean_ms_per_step']:.0f} ms/step")
    acc = None
    if "analytical" in summ and summ["analytical"]["n_runs"] >= 20:
        acc = summ["analytical"]["n_success"] >= 18
        print("acceptance (analytical >= 18/20):", "PASS" if acc else "FAIL")
    cfg_path = RES_DIR / "p5_mppi_config.json"
    out = dict(config=json.loads(cfg_path.read_text()) if cfg_path.exists() else "defaults",
               summary=summ, acceptance_analytical_ge_18_of_20=acc,
               runs={n: {str(sd): r["score"] for sd, r in rs.items()} for n, rs in runs.items()})
    (RES_DIR / "p5_closed_loop.json").write_text(json.dumps(out, indent=2))
    print("wrote results/p5_closed_loop.json")

    # ---- success bars with Wilson intervals
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4.6))
    rate = np.array([summ[n]["success_rate"] for n in names])
    lo = np.array([summ[n]["wilson95"][0] for n in names])
    hi = np.array([summ[n]["wilson95"][1] for n in names])
    ax.bar(range(len(names)), rate, color=[style(n)["color"] for n in names], alpha=0.85,
           yerr=[rate - lo, hi - rate], capsize=5)
    for i, n in enumerate(names):
        ax.text(i, min(hi[i] + 0.03, 1.02), f"{summ[n]['n_success']}/{summ[n]['n_runs']}", ha="center")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label(n) for n in names], rotation=20, ha="right")
    ax.set(ylabel="success rate (Wilson 95 % interval)", ylim=(0, 1.12), title="Closed-loop swing-up success")
    fig.tight_layout()
    savefig(fig, "p5_success")

    # ---- fraction balanced vs time (failures are censored at 10 s)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grid = np.linspace(0, EPISODE_S, 401)
    for n in names:
        ttb = np.array([r["score"]["time_to_balance"] for r in runs[n].values() if r["score"]["success"]])
        frac = np.array([(ttb <= t).sum() for t in grid]) / summ[n]["n_runs"]
        ax.step(grid, frac, where="post", lw=2, label=label(n), **style(n))
    ax.set(xlabel="time [s]", ylabel="fraction balanced (held 2 s, rail never hit)", ylim=(0, 1.02),
           title="Time to balance (failures censored at 10 s)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "p5_time_to_balance")

    # ---- trajectories for one seed per model
    common = set.intersection(*[set(runs[n]) for n in names])
    seed = min(common) if common else min(runs[names[0]])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for n in names:
        if seed not in runs[n]:
            continue
        z = np.load(runs[n][seed]["path"])
        t, q, u = z["t"], z["q"], z["u"]
        axes[0].plot(t, np.arctan2(np.sin(q[:, 1]), np.cos(q[:, 1])), label=label(n), **style(n))
        axes[1].plot(t, q[:, 0], **style(n))
        axes[2].plot(t, u, **style(n))
    axes[1].axhline(1.8, color="k", lw=0.8, ls=":")
    axes[1].axhline(-1.8, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel(r"wrapped $\theta$ [rad]")
    axes[1].set_ylabel("cart $x$ [m] (dotted: rail)")
    axes[2].set(ylabel="$u$", xlabel="time [s]")
    axes[0].set_title(f"Swing-up, env seed {seed}")
    axes[0].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    savefig(fig, "p5_trajectories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "run", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--stress", action="store_true", help="also run the regime-A20 models")
    ap.add_argument("--all-seeds", action="store_true", help="all 3 training seeds of each learned model")
    ap.add_argument("--default-config", action="store_true", help="ignore results/p5_mppi_config.json")
    args = ap.parse_args()
    if args.stage in ("all", "run"):
        stage_run(args)
    if args.stage in ("all", "report"):
        stage_report(args)


if __name__ == "__main__":
    main()
