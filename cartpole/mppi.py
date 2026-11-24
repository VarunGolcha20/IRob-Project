"""Information-theoretic MPPI (Williams et al. 2017) for the cart-pole swing-up (PLAN 5.1).

One call = one control step at 100 Hz:
  1. K noise sequences eps_k in R^H, AR(1)-filtered (coefficient `ar`) for smooth forces;
  2. U_k = clip(U + eps_k, -1, 1);
  3. batched RK4 rollouts with dt_plan, tau = [gear u, 0];
  4. S_k = sum_t c(s_t, u_t) + terminal_mult * c_state(s_H)  (+ the control-cost term c_k below);
  5. weights w_k = softmax(-S_k / lam - ctrl_cost * c_k), c_k = sum_t U_t eps_kt / sigma^2,
     with lam found by bisection so that ESS = 1 / sum w^2 lies in [ess_lo K, ess_hi K].
     ctrl_cost = 0 by default (deviation from the PLAN 5.1 formula; see MPPIConfig.ctrl_cost);
  6. U <- clip(U + sum_k w_k eps_k); apply U[0].
The plan is stored in units of dt_plan, so it is shifted by one element every
dt_plan / dt_control calls (warm start).

Optional LQR hand-over near upright (hysteresis), with the gain computed from
the SAME model's autograd Jacobian at the upright, so the controller depends
only on the model under test.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from .integrators import rk4_step


@dataclass
class MPPIConfig:
    K: int = 512
    H: int = 75
    dt_plan: float = 0.02
    dt_control: float = 0.01
    sigma: float = 0.5                 # noise std in u units (0.5 -> 5 N)
    ar: float = 0.8                    # AR(1) coefficient of the noise
    ess_lo: float = 0.10
    ess_hi: float = 0.20
    # stage cost: w_theta (1 - cos th) + w_x x^2 + w_xd xd^2 + w_thd thd^2 + w_u u^2
    w_theta: float = 10.0
    w_x: float = 1.0                   # 1 / (1 m)^2
    w_xd: float = 0.25                 # 1 / (2 m/s)^2
    w_thd: float = 0.04                # 1 / (5 rad/s)^2
    w_u: float = 0.1
    barrier_w: float = 1e3
    barrier_x: float = 1.6             # 0.2 m margin to the 1.8 m rail
    terminal_mult: float = 10.0
    w_energy: float = 0.0              # energy shaping: w_E ((E_p - m_p g l) / (m_p g l))^2
    # scale of the control-cost term sum_t U_t eps_t / sigma^2 in the weights. The textbook value is 1, but
    # with H = 75 and sigma = 0.5 that term has a spread of ~17 in the weight logits, which the temperature
    # cannot shrink: it dominated the trajectory cost and the swing-up never started (0/20 even for the
    # analytical model, ISSUES C1). 0 drops it, as most practical MPPI implementations do.
    ctrl_cost: float = 0.0
    # LQR hand-over
    use_lqr: bool = True
    lqr_in_th: float = 0.3
    lqr_in_thd: float = 2.0
    lqr_out_th: float = 0.6
    lqr_out_thd: float = 4.0
    bryson: tuple = (1.0, 0.3, 2.0, 2.0)     # same weights as the balance pool S

    @classmethod
    def from_dict(cls, d: dict) -> "MPPIConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        d = {k: (tuple(v) if k == "bryson" else v) for k, v in d.items() if k in names}
        return cls(**d)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def lqr_gain(model, p, bryson=(1.0, 0.3, 2.0, 2.0)):
    """K such that u = -K (x, th, xd, thd) stabilises the upright of `model`.

    A, B come from the autograd Jacobian of the model's own qdd at the upright
    (q = 0, qd = 0, u = 0). Returns None if the model's linearisation is not
    stabilisable or the closed loop is not stable (e.g. a model that never saw
    the upright region).
    """
    from scipy.linalg import solve_continuous_are

    def f(s, u):
        q, qd = s[:2], s[2:]
        tau = torch.stack([p.gear * u[0], u.new_zeros(())])
        return torch.cat([qd, model.qdd(q, qd, tau)])

    s0 = torch.zeros(4, dtype=torch.float64)
    u0 = torch.zeros(1, dtype=torch.float64)
    try:
        Js, Ju = torch.autograd.functional.jacobian(f, (s0, u0))
        A, B = Js.numpy().astype(float), Ju.numpy().astype(float)
        Q = np.diag(1.0 / np.asarray(bryson) ** 2)
        R = np.eye(1)
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P).ravel()
        eig = np.linalg.eigvals(A - B @ K[None, :])
        if not np.all(np.isfinite(K)) or np.any(eig.real >= 0):
            return None
        return K
    except Exception:
        return None


class MPPIController:
    def __init__(self, model, p, cfg: MPPIConfig, seed: int = 0):
        self.model, self.p, self.cfg = model, p, cfg
        self.agl = p.mpl * p.g
        self.gen = torch.Generator().manual_seed(int(seed))
        self.n_sub = max(1, round(cfg.dt_plan / cfg.dt_control))
        self.K_lqr = lqr_gain(model, p, cfg.bryson) if cfg.use_lqr else None
        self.reset()

    def reset(self):
        self.U = torch.zeros(self.cfg.H, dtype=torch.float64)
        self.calls = 0
        self.lqr_active = False
        self.n_switch = 0
        self.last_ess = float("nan")
        self.last_lam = float("nan")

    # ---- costs -------------------------------------------------------------
    def _state_cost(self, q, qd):
        c = self.cfg
        x, th, xd, thd = q[:, 0], q[:, 1], qd[:, 0], qd[:, 1]
        cost = (c.w_theta * (1.0 - torch.cos(th)) + c.w_x * x ** 2 + c.w_xd * xd ** 2
                + c.w_thd * thd ** 2 + c.barrier_w * torch.relu(x.abs() - c.barrier_x) ** 2)
        if c.w_energy > 0:
            E = 0.5 * self.p.J * thd ** 2 + self.agl * torch.cos(th)
            cost = cost + c.w_energy * ((E - self.agl) / self.agl) ** 2
        return cost

    def _noise(self):
        c = self.cfg
        n = torch.randn(c.K, c.H, generator=self.gen, dtype=torch.float64) * c.sigma
        out = torch.empty_like(n)
        out[:, 0] = n[:, 0]
        s = math.sqrt(1.0 - c.ar ** 2)
        for t in range(1, c.H):
            out[:, t] = c.ar * out[:, t - 1] + s * n[:, t]
        return out

    def _weights(self, S, ctrl):
        """softmax weights with lam chosen so that ESS is in [ess_lo K, ess_hi K]."""
        c = self.cfg
        S0 = S - S.min()
        lo, hi = math.log(1e-4), math.log(1e7)
        w, lam = None, 1.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            lam = math.exp(mid)
            logits = -S0 / lam - ctrl
            w = torch.softmax(logits, 0)
            ess = float(1.0 / (w ** 2).sum())
            if ess < c.ess_lo * c.K:
                lo = mid              # too peaked: raise the temperature
            elif ess > c.ess_hi * c.K:
                hi = mid
            else:
                break
        self.last_ess, self.last_lam = ess, lam
        return w

    # ---- planning ----------------------------------------------------------
    @torch.no_grad()
    def plan(self, q, qd):
        c, p = self.cfg, self.p
        noise = self._noise()                                     # (K, H)
        Uk = (self.U[None, :] + noise).clamp(-1.0, 1.0)
        qs = torch.as_tensor(q, dtype=torch.float64).expand(c.K, 2).clone()
        qds = torch.as_tensor(qd, dtype=torch.float64).expand(c.K, 2).clone()
        S = torch.zeros(c.K, dtype=torch.float64)
        for t in range(c.H):
            u = Uk[:, t]
            S = S + self._state_cost(qs, qds) + c.w_u * u ** 2
            tau = torch.stack([p.gear * u, torch.zeros_like(u)], -1)
            qs, qds = rk4_step(self.model, qs, qds, tau, c.dt_plan)
        S = S + c.terminal_mult * self._state_cost(qs, qds)
        S = torch.nan_to_num(S, nan=1e12, posinf=1e12, neginf=1e12)    # diverged rollouts are bad
        ctrl = c.ctrl_cost * (self.U[None, :] * noise).sum(1) / c.sigma ** 2
        w = self._weights(S, ctrl)
        self.U = (self.U + (w[:, None] * noise).sum(0)).clamp(-1.0, 1.0)

    def _shift(self):
        self.calls += 1
        if self.calls % self.n_sub == 0:
            self.U = torch.roll(self.U, -1)
            self.U[-1] = self.U[-2]

    def __call__(self, q, qd) -> float:
        c = self.cfg
        th = wrap(float(q[1]))
        if self.K_lqr is not None:
            if self.lqr_active:
                if abs(th) > c.lqr_out_th or abs(float(qd[1])) > c.lqr_out_thd:
                    self.lqr_active = False
            elif abs(th) < c.lqr_in_th and abs(float(qd[1])) < c.lqr_in_thd and abs(float(q[0])) < c.barrier_x:
                self.lqr_active = True
                self.n_switch += 1
        if self.lqr_active:
            s = np.array([q[0], th, qd[0], qd[1]], dtype=float)
            self._shift()
            return float(np.clip(-self.K_lqr @ s, -1.0, 1.0))
        self.plan(q, qd)
        u0 = float(self.U[0])
        self._shift()
        return u0
