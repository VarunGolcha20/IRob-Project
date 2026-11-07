"""Batched RK4 integration of a model with interface qdd(q, qd, tau).

The force tau is held constant over a step (zero-order hold), exactly as MuJoCo
holds data.ctrl constant across its four RK4 stages.
"""
from __future__ import annotations

import math

import torch


def rk4_step(model, q, qd, tau, h: float):
    def f(q_, qd_):
        return qd_, model.qdd(q_, qd_, tau)

    k1q, k1v = f(q, qd)
    k2q, k2v = f(q + 0.5 * h * k1q, qd + 0.5 * h * k1v)
    k3q, k3v = f(q + 0.5 * h * k2q, qd + 0.5 * h * k2v)
    k4q, k4v = f(q + h * k3q, qd + h * k3v)
    q_next = q + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
    qd_next = qd + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return q_next, qd_next


def rollout(model, q0, qd0, tau_seq, h: float):
    """Open-loop rollout.

    q0, qd0: (..., 2); tau_seq: (T, ..., 2) applied one per step.
    Returns q, qd of shape (T + 1, ..., 2), including the initial state.
    """
    qs, qds = [q0], [qd0]
    q, qd = q0, qd0
    for t in range(tau_seq.shape[0]):
        q, qd = rk4_step(model, q, qd, tau_seq[t], h)
        qs.append(q)
        qds.append(qd)
    return torch.stack(qs), torch.stack(qds)


def wrap_angle(a):
    """Map angles to (-pi, pi]."""
    if isinstance(a, torch.Tensor):
        return torch.atan2(torch.sin(a), torch.cos(a))
    import numpy as np
    return np.arctan2(np.sin(a), np.cos(a))


TWO_PI = 2 * math.pi
