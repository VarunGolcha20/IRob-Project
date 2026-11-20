"""Deep Lagrangian Network (DeLaN) for the cart-pole.

Learns L = 1/2 qd^T M(th) qd - V(th) and computes everything else analytically
(docs/PART3_MODELS.md):
    M = L_ch L_ch^T + eps I        (L_ch lower-triangular, positive diagonal)
    C qd = Mdot qd - 1/2 d/dq (qd^T M qd)        (Coriolis/centripetal)
    g = dV/dq
    qdd = M^-1 (tau - C qd - g - D qd),  D = diag(softplus(d)) >= 0

M and V depend on th only (translational symmetry, PART1 P2), so every
q-derivative is a th-derivative. It is computed by forward-mode
differentiation through the networks: each hidden layer carries
(h, dh/dth), with
    Linear:   z = W h + b,          dz = W dh
    softplus: a = softplus(z),      da = sigmoid(z) * dz
This costs one extra matrix product per layer, works under torch.no_grad()
(for MPPI), and is ordinary differentiable torch code for training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..analytical import solve2


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class DualMLP(nn.Module):
    """MLP on (sin th, cos th) that also returns d(output)/d(th)."""

    def __init__(self, n_out: int, hidden: int, depth: int):
        super().__init__()
        dims = [2] + [hidden] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], n_out)

    def forward(self, th: torch.Tensor):
        s, c = torch.sin(th), torch.cos(th)
        h = torch.stack([s, c], -1)
        dh = torch.stack([c, -s], -1)            # d/dth (sin th, cos th)
        for lin in self.hidden:
            z = lin(h)
            dz = dh @ lin.weight.T
            h, dh = F.softplus(z), torch.sigmoid(z) * dz
        return self.out(h), dh @ self.out.weight.T


def lagrangian_dynamics(M11, M12, M22, dM11, dM12, dM22, dV, d, q, qd, tau):
    """Forward dynamics from M(th), V(th) and their th-derivatives.

    Returns qdd and the intermediate terms (C qd, g, D qd). All inputs have a
    leading batch shape; q, qd, tau end in 2.
    """
    xd, thd = qd[..., 0], qd[..., 1]
    # Mdot qd = (dM/dth) thd qd
    dMqd0 = dM11 * xd + dM12 * thd
    dMqd1 = dM12 * xd + dM22 * thd
    # 1/2 d/dth (qd^T M qd) = 1/2 qd^T (dM/dth) qd ; d/dx (...) = 0
    half_dT = 0.5 * (dM11 * xd ** 2 + 2 * dM12 * xd * thd + dM22 * thd ** 2)
    Cqd = torch.stack([thd * dMqd0, thd * dMqd1 - half_dT], -1)
    g = torch.stack([torch.zeros_like(dV), dV], -1)
    Dqd = d * qd
    M = torch.stack([torch.stack([M11, M12], -1), torch.stack([M12, M22], -1)], -2)
    qdd = solve2(M, tau - Cqd - g - Dqd)
    return qdd, dict(M=M, Cqd=Cqd, g=g, Dqd=Dqd)


class DeLaN(nn.Module):
    name = "delan"

    def __init__(self, hidden: int = 128, depth: int = 2, eps: float = 1e-3,
                 damping_init: float = 1e-4):
        super().__init__()
        self.config = dict(hidden=hidden, depth=depth, eps=eps, damping_init=damping_init)
        self.eps = eps
        self.L_net = DualMLP(3, hidden, depth)     # (l11_raw, l21, l22_raw)
        self.V_net = DualMLP(1, hidden, depth)
        # ISSUES P1: softplus(0) = 0.69 would mean a 1.6 s damping time constant.
        self.d_raw = nn.Parameter(torch.full((2,), inv_softplus(damping_init)))
        self.damping_on = True                     # switched off for the undamped drift test

    # ---- structure ----------------------------------------------------------
    def structure(self, th: torch.Tensor):
        """M entries, V and their th-derivatives at angles th (any shape)."""
        shape = th.shape
        th = th.reshape(-1)
        l, dl = self.L_net(th)
        V, dV = self.V_net(th)
        r11, l21, r22 = l.unbind(-1)
        dr11, dl21, dr22 = dl.unbind(-1)
        l11, l22 = F.softplus(r11), F.softplus(r22)
        dl11, dl22 = torch.sigmoid(r11) * dr11, torch.sigmoid(r22) * dr22
        out = dict(
            M11=l11 ** 2 + self.eps,
            M12=l11 * l21,
            M22=l21 ** 2 + l22 ** 2 + self.eps,
            dM11=2 * l11 * dl11,
            dM12=dl11 * l21 + l11 * dl21,
            dM22=2 * l21 * dl21 + 2 * l22 * dl22,
            V=V[:, 0], dV=dV[:, 0],
        )
        return {k: v.reshape(shape) for k, v in out.items()}

    @property
    def damping(self) -> torch.Tensor:
        return F.softplus(self.d_raw)

    def _d(self, like):
        if not self.damping_on:
            return torch.zeros(2, dtype=like.dtype, device=like.device)
        return self.damping.to(like.dtype)

    # ---- dynamics (common interface) ------------------------------------------
    def qdd(self, q, qd, tau):
        return self.forward_terms(q, qd, tau)[0]

    def forward_terms(self, q, qd, tau):
        s = self.structure(q[..., 1])
        return lagrangian_dynamics(s["M11"], s["M12"], s["M22"], s["dM11"], s["dM12"], s["dM22"],
                                   s["dV"], self._d(q), q, qd, tau)

    def inverse(self, q, qd, qdd):
        """tau = M qdd + C qd + g + D qd."""
        s = self.structure(q[..., 1])
        zero = torch.zeros_like(qd)
        _, t = lagrangian_dynamics(s["M11"], s["M12"], s["M22"], s["dM11"], s["dM12"], s["dM22"],
                                   s["dV"], self._d(q), q, qd, zero)
        Mqdd = (t["M"] @ qdd.unsqueeze(-1)).squeeze(-1)
        return Mqdd + t["Cqd"] + t["g"] + t["Dqd"], s

    def energy(self, q, qd):
        s = self.structure(q[..., 1])
        xd, thd = qd[..., 0], qd[..., 1]
        T = 0.5 * (s["M11"] * xd ** 2 + 2 * s["M12"] * xd * thd + s["M22"] * thd ** 2)
        return T + s["V"]
