"""Phase 3: train every model configuration.

Stage "select": choose DeLaN's inverse-loss weight lam_inv in {0, 0.1, 1} on
validation (regimes A20 and B20, seed 0), picking the lowest mean validation
loss. Stage "main": the full grid with that lam_inv.
    regime A: N = 1 (seeds 0-4), N = 5 (seeds 0-2), N = 20 (seeds 0-2)
    regime B: N = 5, 20 (seeds 0-2)
    both models -> 34 runs
Runs execute in parallel worker processes, each with 2 torch threads.

Usage:  python scripts/p3_train.py [--stage select|main|all] [--workers 4] [--steps 30000]
Outputs: weights/*.pt, results/p3_select.json, results/p3_training.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cartpole.plotting import RES_DIR  # noqa: E402

LAMS = (0.0, 0.1, 1.0)
GRID = [("A", 1, range(5)), ("A", 5, range(3)), ("A", 20, range(3)),
        ("B", 5, range(3)), ("B", 20, range(3))]


def _worker(cfg):
    from cartpole.train import train_one
    t = time.time()
    r = train_one(cfg)
    print(f"done {r['name']:<22} best_val {r['best_val']:.3e} at {r['best_step']:>5} "
          f"(stopped {r['stopped_step']:>5}) {time.time() - t:.0f}s", flush=True)
    return r


def run_all(cfgs, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(_worker, cfgs, chunksize=1)


def stage_select(args):
    cfgs = [dict(kind="delan", regime=reg, N=20, seed=0, steps=args.steps, lam_inv=lam)
            for reg in ("A", "B") for lam in LAMS]
    res = run_all(cfgs, args.workers)
    table = {f"{r['cfg']['regime']}20_lam{r['cfg']['lam_inv']:g}": r["best_val"] for r in res}
    mean = {lam: sum(r["best_val"] for r in res if r["cfg"]["lam_inv"] == lam) / 2 for lam in LAMS}
    best = min(mean, key=mean.get)
    out = dict(val_loss=table, mean_over_regimes={f"{k:g}": v for k, v in mean.items()}, chosen_lam_inv=best)
    (RES_DIR / "p3_select.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return best


def stage_main(args, lam):
    cfgs = []
    for reg, N, seeds in GRID:
        for s in seeds:
            # longest jobs (DeLaN) first, so the pool finishes evenly
            cfgs.append(dict(kind="delan", regime=reg, N=N, seed=s, steps=args.steps, lam_inv=lam))
    for reg, N, seeds in GRID:
        for s in seeds:
            cfgs.append(dict(kind="mlp", regime=reg, N=N, seed=s, steps=args.steps))
    # lam > 0 runs from the selection stage with seed 0 at N = 20 are reused.
    from cartpole.train import run_name
    from cartpole.plotting import WEIGHT_DIR
    todo = [c for c in cfgs if not (WEIGHT_DIR / f"{run_name(c)}.pt").exists() or args.retrain]
    res = run_all(todo, args.workers) if todo else []
    print(f"{len(res)} runs trained, {len(cfgs) - len(todo)} reused")


def summarise(lam):
    import torch
    from cartpole.plotting import WEIGHT_DIR
    from cartpole.train import run_name
    rows = []
    for reg, N, seeds in GRID:
        for s in seeds:
            for kind in ("mlp", "delan"):
                c = dict(kind=kind, regime=reg, N=N, seed=s, lam_inv=lam if kind == "delan" else 0.0)
                ck = torch.load(WEIGHT_DIR / f"{run_name(c)}.pt", weights_only=False)
                rows.append(dict(name=ck["name"], kind=kind, regime=reg, N=N, seed=s,
                                 best_val=ck["best_val"], best_step=ck["best_step"],
                                 stopped_step=ck["stopped_step"], time_s=ck["time_s"],
                                 n_train=ck["n_train"], n_params=ck["n_params"], chosen=ck["chosen"]))
    (RES_DIR / "p3_training.json").write_text(json.dumps(dict(lam_inv=lam, runs=rows), indent=2))
    print(f"wrote {len(rows)} runs to results/p3_training.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select", "main", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("select", "all"):
        lam = stage_select(args)
    else:
        lam = json.loads((RES_DIR / "p3_select.json").read_text())["chosen_lam_inv"]
    if args.stage in ("main", "all"):
        stage_main(args, lam)
        summarise(lam)


if __name__ == "__main__":
    main()
