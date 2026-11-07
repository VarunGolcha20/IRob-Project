"""Data collection from dm_control cartpole.

Row t of an episode holds the state s_t, the action u_t applied from it, and
every MuJoCo quantity at that instant (after mj_forward with ctrl = u_t):
    s_{t+1} = RK4(s_t, u_t)      (MuJoCo's integrator, dt = 0.01 s)
q_ddot is MuJoCo's continuous-time acceleration, not a finite difference.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

FIELDS = (
    "t", "qpos", "qvel", "u", "qfrc_actuator", "qfrc_constraint",
    "qfrc_passive", "qfrc_bias", "qacc", "fullM", "energy", "limit_active",
)


def snapshot(physics, u: float, t: float) -> dict:
    """Set ctrl = u, recompute the dynamics at the current state, read it out."""
    physics.data.ctrl[:] = u
    physics.forward()
    m, d = physics.model.ptr, physics.data.ptr
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)  # MuJoCo >= 4.0 signature
    return dict(
        t=t,
        qpos=d.qpos.copy(),
        qvel=d.qvel.copy(),
        u=float(u),
        qfrc_actuator=d.qfrc_actuator.copy(),
        qfrc_constraint=d.qfrc_constraint.copy(),
        qfrc_passive=d.qfrc_passive.copy(),
        qfrc_bias=d.qfrc_bias.copy(),
        qacc=d.qacc.copy(),
        fullM=M,
        energy=d.energy.copy(),
        limit_active=bool(d.nefc > 0),
    )


def run_episode(env, policy, n_steps: int | None = None, physics_hook=None) -> dict:
    """Run one episode with policy(physics, t_index) -> u in [-1, 1].

    physics_hook(physics) is called after reset (e.g. to zero damping or set a
    custom initial state). Returns a dict of stacked arrays (one row per step).
    """
    ts = env.reset()
    physics = env.physics
    if physics_hook is not None:
        physics_hook(physics)
    rows = []
    k = 0
    while not ts.last() and (n_steps is None or k < n_steps):
        u = float(np.clip(policy(physics, k), -1.0, 1.0))
        rows.append(snapshot(physics, u, physics.data.time))
        ts = env.step(u)
        k += 1
    ep = {f: np.array([r[f] for r in rows]) for f in FIELDS}
    # Final state (no action applied from it) so rollouts can be checked to the end.
    ep["qpos_final"] = physics.data.qpos.copy()
    ep["qvel_final"] = physics.data.qvel.copy()
    return ep


def random_policy(seed: int):
    """u ~ U[-1, 1], independent at every step (the PDF's excitation)."""
    rng = np.random.default_rng(seed)
    return lambda physics, k: rng.uniform(-1.0, 1.0)


def save_episodes(path: Path, episodes: list[dict], meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for i, ep in enumerate(episodes):
        for k, v in ep.items():
            flat[f"ep{i:03d}/{k}"] = v
    flat["n_episodes"] = np.array(len(episodes))
    if meta:
        for k, v in meta.items():
            flat[f"meta/{k}"] = np.asarray(v)
    np.savez_compressed(path, **flat)


def load_episodes(path: Path) -> tuple[list[dict], dict]:
    z = np.load(path)
    n = int(z["n_episodes"])
    eps = [dict() for _ in range(n)]
    meta = {}
    for key in z.files:
        if key.startswith("ep"):
            i, field = key.split("/", 1)
            eps[int(i[2:])][field] = z[key]
        elif key.startswith("meta/"):
            meta[key[5:]] = z[key]
    return eps, meta
