"""Model loading shared by Phases 4-6.

A model is named by a *spec string*:
    "analytical"      exact EL model (MuJoCo parameters)
    "thin_rod"        textbook thin-rod pole inertia
    "analytical_p1"   exact model with every dynamics parameter +1 %
    "<path>.pt"       a trained MLP or DeLaN checkpoint (float64, eval mode)
Strings (not objects) are passed to worker processes, which load the model themselves.
"""
from __future__ import annotations

import dataclasses
import json
from functools import lru_cache

from .analytical import AnalyticalModel, thin_rod_model
from .params import load_params
from .plotting import RES_DIR, WEIGHT_DIR

ANALYTIC = ("analytical", "thin_rod", "analytical_p1")
P1_FIELDS = ("m_c", "m_p", "l", "I_c", "b_x", "b_th", "gear", "g")   # parameters that enter the dynamics


@lru_cache(maxsize=None)
def get_params():
    return load_params()


def load_any(spec: str):
    p = get_params()
    if spec == "analytical":
        return AnalyticalModel(p)
    if spec == "thin_rod":
        return thin_rod_model(p)
    if spec == "analytical_p1":
        return AnalyticalModel(dataclasses.replace(p, **{f: getattr(p, f) * 1.01 for f in P1_FIELDS}))
    from .train import load_model          # lazy: train.py imports matplotlib (Agg backend)
    model, _ = load_model(spec)
    return model


def training_info() -> dict:
    return json.loads((RES_DIR / "p3_training.json").read_text())


def grid_runs() -> list[dict]:
    """Every trained run of the Phase 3 grid, with its weight path."""
    return [dict(kind=r["kind"], regime=r["regime"], N=r["N"], seed=r["seed"], name=r["name"],
                 best_val=r["best_val"], path=str(WEIGHT_DIR / f"{r['name']}.pt"))
            for r in training_info()["runs"]]


def best_run(kind: str, regime: str, N: int) -> dict:
    """The seed with the lowest *validation* loss (never a test metric)."""
    rows = [r for r in grid_runs() if r["kind"] == kind and r["regime"] == regime and r["N"] == N]
    return min(rows, key=lambda r: r["best_val"])
