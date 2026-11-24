"""Closed-loop MPPI episodes in dm_control cartpole/swingup, and their scoring (PLAN 5.3).

Balanced:        |wrap th| < 0.2 and |thd| < 1.
Time-to-balance: first t at which "balanced" then holds for 2 s (inside the 10 s episode).
Success:         a time-to-balance exists AND the rail limit was never active.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

BAL_TH, BAL_THD, HOLD_S = 0.2, 1.0, 2.0


def wrap_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def score(res: dict, dt: float = 0.01) -> dict:
    q, qd, t = res["q"], res["qd"], res["t"]
    n = len(t)
    bal = (np.abs(wrap_np(q[:, 1])) < BAL_TH) & (np.abs(qd[:, 1]) < BAL_THD)
    need = int(round(HOLD_S / dt))
    run = np.zeros(n + 1, int)
    for i in range(n - 1, -1, -1):
        run[i] = run[i + 1] + 1 if bal[i] else 0
    idx = np.where(run[:n] >= need)[0]
    ttb = float(t[idx[0]]) if len(idx) else None
    limit_hit = bool(res["limit"].any())
    return dict(success=bool(ttb is not None and not limit_hit), time_to_balance=ttb,
                limit_hit=limit_hit, max_abs_x=float(np.abs(q[:, 0]).max()),
                final_abs_theta=float(abs(wrap_np(q[-1, 1]))), ret=float(res["reward"].sum()),
                ms_per_step=float(1e3 * np.mean(res["wall_s"])), lqr_steps=int(res["lqr"].sum()),
                lqr_switches=int(res["n_switch"]))


def run_swingup(model, cfg, seed: int, seconds: float = 10.0, hook=None) -> dict:
    """One closed-loop episode. `hook(physics, t, u)` (optional) is called every step."""
    from .loading import get_params
    from .mppi import MPPIController
    from .params import make_env

    p = get_params()
    env = make_env(seed)
    ts = env.reset()
    physics = env.physics
    ctrl = MPPIController(model, p, cfg, seed)
    rows = dict(t=[], q=[], qd=[], u=[], limit=[], lqr=[], reward=[], wall_s=[])
    while not ts.last() and physics.data.time < seconds - 1e-9:
        q, qd = physics.data.qpos.copy(), physics.data.qvel.copy()
        t0 = time.perf_counter()
        u = ctrl(q, qd)
        wall = time.perf_counter() - t0
        rows["t"].append(float(physics.data.time))
        rows["q"].append(q)
        rows["qd"].append(qd)
        rows["u"].append(u)
        rows["lqr"].append(ctrl.lqr_active)
        rows["wall_s"].append(wall)
        if hook is not None:
            hook(physics, rows["t"][-1], u)
        ts = env.step(np.array([u]))
        rows["limit"].append(bool(physics.data.nefc > 0))
        rows["reward"].append(0.0 if ts.reward is None else float(ts.reward))
    res = {k: np.asarray(v) for k, v in rows.items()}
    res["n_switch"] = ctrl.n_switch
    res["score"] = score(res, p.dt)
    return res


def episode_job(job: dict) -> dict:
    """Worker entry point (spawn-safe). job: spec, cfg (dict), seed, name, save_dir (or None).

    Resumable: if the run's .npz already exists, its stored score is returned.
    """
    import torch
    from .loading import load_any
    from .mppi import MPPIConfig

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    path = None
    if job.get("save_dir"):
        path = Path(job["save_dir"]) / f"{job['name']}_seed{job['seed']}.npz"
        if path.exists():
            z = np.load(path)
            return dict(name=job["name"], seed=job["seed"], **json.loads(str(z["score"])))
    model = load_any(job["spec"])
    res = run_swingup(model, MPPIConfig.from_dict(job["cfg"]), job["seed"], job.get("seconds", 10.0))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=res["t"], q=res["q"], qd=res["qd"], u=res["u"], limit=res["limit"],
                            lqr=res["lqr"], reward=res["reward"], score=json.dumps(res["score"]))
    return dict(name=job["name"], seed=job["seed"], **res["score"])


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
