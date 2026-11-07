"""Analytical Euler-Lagrange model of the cart-pole (batched, torch).

Coordinates q = (x, theta), theta = 0 upright, theta = pi hanging.
    T = 1/2 (m_c+m_p) xd^2 + m_p l cos(th) xd thd + 1/2 J thd^2
    V = m_p g l cos(th)            (+ constant, see energy())
    M(th) qdd + C(q,qd) qd + g(q) + D qd = tau,   tau = [gear*u (+ constraint), 0]

All functions accept tensors with a trailing dimension of 2 and any batch shape.
"""
from __future__ import annotations

import dataclasses

import torch

from .params import CartpoleParams


class AnalyticalModel:
    """Exact cart-pole dynamics. Implements the common model interface
    qdd(q, qd, tau) used by the integrator and MPPI."""

    name = "analytical"

    def __init__(self, p: CartpoleParams, damping: bool = True):
        self.p = p
        self.damping = damping

    # ---- structure -------------------------------------------------------
    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        p = self.p
        c = torch.cos(q[..., 1])
        M = torch.empty(q.shape[:-1] + (2, 2), dtype=q.dtype, device=q.device)
        M[..., 0, 0] = p.M_tot
        M[..., 0, 1] = p.mpl * c
        M[..., 1, 0] = p.mpl * c
        M[..., 1, 1] = p.J
        return M

    def coriolis(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """C(q, qd) qd."""
        s = torch.sin(q[..., 1])
        thd = qd[..., 1]
        return torch.stack([-self.p.mpl * s * thd ** 2, torch.zeros_like(thd)], -1)

    def gravity(self, q: torch.Tensor) -> torch.Tensor:
        """g(q) = dV/dq."""
        s = torch.sin(q[..., 1])
        return torch.stack([torch.zeros_like(s), -self.p.mpl * self.p.g * s], -1)

    def damping_force(self, qd: torch.Tensor) -> torch.Tensor:
        """D qd (zero if damping disabled)."""
        if not self.damping:
            return torch.zeros_like(qd)
        return torch.stack([self.p.b_x * qd[..., 0], self.p.b_th * qd[..., 1]], -1)

    def bias(self, q, qd):
        """C qd + g  (MuJoCo's qfrc_bias)."""
        return self.coriolis(q, qd) + self.gravity(q)

    # ---- dynamics --------------------------------------------------------
    def qdd(self, q, qd, tau):
        M = self.mass_matrix(q)
        rhs = tau - self.bias(q, qd) - self.damping_force(qd)
        return solve2(M, rhs)

    def inverse(self, q, qd, qdd):
        M = self.mass_matrix(q)
        return (M @ qdd.unsqueeze(-1)).squeeze(-1) + self.bias(q, qd) + self.damping_force(qd)

    # ---- energy ----------------------------------------------------------
    def kinetic(self, q, qd):
        M = self.mass_matrix(q)
        return 0.5 * (qd.unsqueeze(-2) @ M @ qd.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def potential(self, q, mujoco_offset: bool = False):
        p = self.p
        V = p.mpl * p.g * torch.cos(q[..., 1])
        if mujoco_offset:  # MuJoCo measures height from z = 0 for every body
            V = V + p.M_tot * p.g * p.z_cart
        return V

    def energy(self, q, qd, mujoco_offset: bool = False):
        return self.kinetic(q, qd) + self.potential(q, mujoco_offset)

    def momentum_x(self, q, qd):
        """p_x = dL/dxd, conserved when tau = 0 and b_x = 0 (Noether)."""
        return self.p.M_tot * qd[..., 0] + self.p.mpl * torch.cos(q[..., 1]) * qd[..., 1]

    # ---- variants --------------------------------------------------------
    def with_params(self, **changes) -> "AnalyticalModel":
        m = AnalyticalModel(dataclasses.replace(self.p, **changes), self.damping)
        return m


def solve2(M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Closed-form solve of a batch of 2x2 systems M x = b (Cramer's rule)."""
    a, b_, c, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    det = a * d - b_ * c
    x0 = (d * b[..., 0] - b_ * b[..., 1]) / det
    x1 = (a * b[..., 1] - c * b[..., 0]) / det
    return torch.stack([x0, x1], -1)


def thin_rod_model(p: CartpoleParams) -> AnalyticalModel:
    """Textbook model: pole as a thin rod of length 2l (I_c = m (2l)^2 / 12),
    i.e. ignoring the capsule radius and hemispherical caps."""
    return AnalyticalModel(dataclasses.replace(p, I_c=p.m_p * (2 * p.l) ** 2 / 12.0))
