"""Training/validation sets for the two regimes (PLAN Phase 3.1, PART2 section 7).

Regime A (random only):        N episodes of R_train.
Regime B (random+pump+balance): N of R_train, N of E_train, ceil(N/2) of S_train.
Validation: R_val (A) or R_val + E_val + S_val (B).
Rail-limit rows are always dropped (docs/ISSUES.md D2).

The episode subset is drawn with a generator seeded by (seed, N, regime), so
each training seed sees a different subset (ISSUES D6) and the draw does not
depend on anything else.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from .data import flatten, load_episodes
from .plotting import DATA_DIR

REGIME_CODE = {"A": 0, "B": 1}


@lru_cache(maxsize=None)
def _pool(split: str):
    eps, _ = load_episodes(DATA_DIR / f"{split}.npz")
    return eps


def _pick(split: str, n: int, rng: np.random.Generator):
    eps = _pool(split)
    idx = rng.choice(len(eps), size=n, replace=False)
    return [eps[i] for i in sorted(idx)], sorted(int(i) for i in idx)


def features(q: torch.Tensor, qd: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """MLP input (x, sin th, cos th, xd, thd, tau_x). tau_th is always 0 (unactuated)."""
    th = q[..., 1]
    return torch.stack([q[..., 0], torch.sin(th), torch.cos(th), qd[..., 0], qd[..., 1], tau[..., 0]], -1)


def _to_torch(d: dict, dtype=torch.float32) -> dict:
    return {k: torch.as_tensor(d[k], dtype=dtype) for k in ("q", "qd", "tau", "qdd")}


def build(regime: str, N: int, seed: int, dtype=torch.float32) -> dict:
    rng = np.random.default_rng([seed, N, REGIME_CODE[regime]])
    chosen = {}
    eps = []
    r_eps, chosen["R_train"] = _pick("R_train", N, rng)
    eps += r_eps
    val = list(_pool("R_val"))
    if regime == "B":
        e_eps, chosen["E_train"] = _pick("E_train", N, rng)
        s_eps, chosen["S_train"] = _pick("S_train", math.ceil(N / 2), rng)
        eps += e_eps + s_eps
        val += list(_pool("E_val")) + list(_pool("S_val"))
    tr = _to_torch(flatten(eps, drop_limit=True), dtype)
    va = _to_torch(flatten(val, drop_limit=True), dtype)

    feats = features(tr["q"], tr["qd"], tr["tau"])
    stats = {
        "in_mean": feats.mean(0), "in_std": feats.std(0),
        "qdd_mean": tr["qdd"].mean(0), "qdd_std": tr["qdd"].std(0),
        "tau_std": tr["tau"][:, 0].std(),
    }
    return dict(train=tr, val=va, stats=stats, chosen=chosen,
                n_train=int(len(tr["q"])), n_val=int(len(va["q"])))
