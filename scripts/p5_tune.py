"""Phase 5 tuning: choose the MPPI configuration with the ANALYTICAL model only (PLAN 5.2).

Protocol (no test-seed leakage):
  * only the analytical model, only env seeds 100-109 (the evaluation uses seeds 0-19);
  * every configuration tried is appended to results/p5_tuning.csv, so nothing is cherry-picked;
  * the winner (most successes, then lowest mean time-to-balance, then lowest cost per step)
    is frozen to results/p5_mppi_config.json, which scripts/p5_mppi.py and scripts/demo.py read.
By default the search stops at the first candidate that succeeds on every tuning seed;
pass --all to try every candidate.

Candidates are overrides of cartpole.mppi.MPPIConfig (the PLAN 5.1 starting settings are the first one).
Edit CANDIDATES to add your own.

Usage:  python scripts/p5_tune.py [--workers 4] [--seeds 100-109] [--all]
Each episode is 1000 control steps of K = 512 rollouts of H = 75 RK4 steps; expect roughly a minute
per episode for the analytical model, so --workers matters.
Outputs: results/p5_tuning.csv, results/p5_mppi_config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cartpole.closedloop import episode_job  # noqa: E402
from cartpole.mppi import MPPIConfig  # noqa: E402
from cartpole.plotting import RES_DIR  # noqa: E402

CANDIDATES = [
    {},                                                        # PLAN 5.1 starting settings
    {"w_energy": 5.0},                                         # + energy shaping
    {"w_theta": 20.0},
    {"sigma": 0.7},
    {"H": 100},
    {"w_theta": 20.0, "w_energy": 5.0},
    {"H": 100, "sigma": 0.7},
    {"H": 100, "sigma": 0.7, "w_energy": 5.0},
    {"H": 100, "sigma": 0.7, "w_theta": 20.0, "w_energy": 5.0},
    {"use_lqr": False},                                        # diagnostic: is the LQR hand-over needed?
    {"ctrl_cost": 1.0},                                        # diagnostic: the textbook weights; 0/10 (ISSUES C1)
]
COLUMNS = ["idx", "overrides", "n_seeds", "n_success", "success_rate", "mean_ttb_s", "n_limit_hits",
           "mean_max_abs_x", "mean_ms_per_step", "per_seed_success", "timestamp"]


def parse_seeds(s: str):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--all", action="store_true", help="try every candidate instead of stopping at the first perfect one")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    assert min(seeds) >= 100, "tuning seeds must not overlap the evaluation seeds 0-19"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RES_DIR / "p5_tuning.csv"
    new = not csv_path.exists()
    start_idx = 0 if new else sum(1 for _ in csv.reader(csv_path.open())) - 1    # trials already logged
    f = csv_path.open("a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(COLUMNS)

    ctx = mp.get_context("spawn")
    best = None
    with ctx.Pool(args.workers) as pool:
        for k, over in enumerate(CANDIDATES):
            cfg = MPPIConfig.from_dict(over).to_dict()
            jobs = [dict(spec="analytical", cfg=cfg, seed=s, name="tune", save_dir=None) for s in seeds]
            t0 = time.time()
            res = pool.map(episode_job, jobs, chunksize=1)
            succ = [r["success"] for r in res]
            ttb = [r["time_to_balance"] for r in res if r["success"]]
            row = dict(
                idx=start_idx + k, overrides=json.dumps(over), n_seeds=len(seeds), n_success=int(sum(succ)),
                success_rate=sum(succ) / len(seeds), mean_ttb_s=float(np.mean(ttb)) if ttb else "",
                n_limit_hits=int(sum(r["limit_hit"] for r in res)),
                mean_max_abs_x=float(np.mean([r["max_abs_x"] for r in res])),
                mean_ms_per_step=float(np.mean([r["ms_per_step"] for r in res])),
                per_seed_success="".join("1" if s else "0" for s in succ),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            w.writerow([row[c] for c in COLUMNS])
            f.flush()
            print(f"[{k + 1}/{len(CANDIDATES)}] {over}: {row['n_success']}/{len(seeds)} success, "
                  f"mean ttb {row['mean_ttb_s']}, {row['mean_ms_per_step']:.1f} ms/step, {time.time() - t0:.0f}s", flush=True)
            key = (-row["n_success"], row["mean_ttb_s"] if ttb else float("inf"), row["mean_ms_per_step"])
            if best is None or key < best[0]:
                best = (key, over, cfg, row)
            if not args.all and row["n_success"] == len(seeds):
                break
    f.close()
    _, over, cfg, row = best
    out = dict(config=cfg, overrides=over, tuning_model="analytical", tuning_seeds=seeds,
               n_success_on_tuning_seeds=row["n_success"], n_tuning_seeds=len(seeds),
               note="Chosen on analytical-model seeds 100-109 only. Log of every trial: results/p5_tuning.csv")
    (RES_DIR / "p5_mppi_config.json").write_text(json.dumps(out, indent=2))
    print(f"frozen: {over} ({row['n_success']}/{len(seeds)}) -> results/p5_mppi_config.json")


if __name__ == "__main__":
    main()
