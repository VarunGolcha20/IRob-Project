"""MPPI swing-up and balance for the cart-pole (PLAN Phase 5, docs/PART5_MPPI.md).

Information-theoretic MPPI (Williams et al. 2017) with
  * AR(1)-correlated exploration noise (smooth forces),
  * temperature lambda set every step by bisection so that the effective sample
    size ESS = 1 / sum w^2 lies in [ess_lo K, ess_hi K],
  * warm start (shift by one planning step),
  * optional LQR hand-over near upright, with the gain computed from the SAME
    model's Jacobian at the upright equilibrium (so every controller depends
    only on the model under test).

The planner works in float32 on K x H batched RK4 rollouts. The plant is the
float64 analytical model + RK4 at h = 0.01 s, which equals MuJoCo off the rail
to 1e-14 (results/p1_validation.json; verified on the test pools in
tests/test_npdyn.py). The rail is not simulated: |x| >= x_limit marks the
episode as a rail failure (the success criterion forbids rail contact anyway).

Timing: the planner step is dt_plan = 0.02 s and the controller replans at
1 / dt_plan = 50 Hz, holding u for 2 simulator steps. This makes the
warm-start shift exactly one planning step (ISSUES C1).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import npdyn as nd


# ---------------------------------------------------------------------------
# fast planning models
# ---------------------------------------------------------------------------
class ThetaTable:
    """DeLaN (or the analytical model) as a periodic lookup table in theta.

    M, dM/dth and dV/dth depend on theta only (translational symmetry), so the
    whole learned model is 7 scalar functions of theta plus the damping. Linear
    interpolation on n points has error <= h^2/8 max|f''|; with n = 8192 this is
    ~1e-7 relative, three orders below DeLaN's own test error (checked in
    tests/test_npdyn.py).
    """

    KEYS = ("M11", "M12", "M22", "dM11", "dM12", "dM22", "dV")

    def __init__(self, model, n: int = 8192, dtype=np.float32, damping=None):
        th = -np.pi + 2 * np.pi * np.arange(n + 1) / n
        s = model.structure(th)
        self.T = np.stack([s[k] for k in self.KEYS], -1).astype(dtype)   # (n+1, 7), last row = first
        self.n, self.dtype = n, dtype
        self.scale = dtype(n / (2 * np.pi))
        if damping is None:
            damping = model.d if isinstance(model, nd.DeLaN) else np.array([model.p.b_x, model.p.b_th])
        self.d = np.asarray(damping, dtype)
        self.name = getattr(model, "name", "table") + "_table"

    def qdd(self, q, qd, tau):
        u = (q[..., 1] + np.pi) * self.scale
        u = u - self.n * np.floor(u / self.n)
        i = np.minimum(u.astype(np.int32), self.n - 1)
        f = (u - i)[..., None]
        v = self.T[i] * (1 - f) + self.T[i + 1] * f
        M11, M12, M22, dM11, dM12, dM22, dV = (v[..., k] for k in range(7))
        xd, thd = qd[..., 0], qd[..., 1]
        c0 = thd * (dM11 * xd + dM12 * thd) + self.d[0] * xd
        c1 = (thd * (dM12 * xd + dM22 * thd)
              - 0.5 * (dM11 * xd * xd + 2 * dM12 * xd * thd + dM22 * thd * thd) + dV + self.d[1] * thd)
        b0, b1 = tau[..., 0] - c0, tau[..., 1] - c1
        det = M11 * M22 - M12 * M12
        out = np.empty(q.shape, self.dtype)
        out[..., 0] = (M22 * b0 - M12 * b1) / det
        out[..., 1] = (M11 * b1 - M12 * b0) / det
        return out


class FastMLP:
    """The MLP baseline in float32 with a cheaper, exact softplus."""

    def __init__(self, m: nd.MLP, dtype=np.float32):
        self.W = [w.astype(dtype) for w in m.W]
        self.b = [b.astype(dtype) for b in m.b]
        self.mu, self.sd = m.in_mean.astype(dtype), m.in_std.astype(dtype)
        self.omu, self.osd = m.out_mean.astype(dtype), m.out_std.astype(dtype)
        # fold input standardisation into the first layer, output de-standardisation into the last
        self.W0 = (self.W[0] / self.sd[:, None]).astype(dtype)
        self.b0 = (self.b[0] - (self.mu / self.sd) @ self.W[0]).astype(dtype)
        self.WL = (self.W[-1] * self.osd[None, :]).astype(dtype)
        self.bL = (self.b[-1] * self.osd + self.omu).astype(dtype)
        self.dtype, self.name = dtype, "mlp_fast"

    @staticmethod
    def _softplus_(z):
        """In place: softplus(z) = max(z, 0) + log1p(exp(-|z|)) (exact, overflow-free)."""
        e = np.abs(z)
        np.negative(e, out=e)
        np.exp(e, out=e)
        np.log1p(e, out=e)
        np.maximum(z, 0, out=z)
        z += e
        return z

    def qdd(self, q, qd, tau):
        th = q[..., 1]
        z = np.stack([q[..., 0], np.sin(th), np.cos(th), qd[..., 0], qd[..., 1], tau[..., 0]], -1)
        h = z @ self.W0
        h += self.b0
        self._softplus_(h)
        for W, b in zip(self.W[1:-1], self.b[1:-1]):
            h = h @ W
            h += b
            self._softplus_(h)
        return h @ self.WL + self.bL


def planning_model(model, dtype=np.float32):
    if isinstance(model, (nd.DeLaN, nd.Analytical)):
        return ThetaTable(model, dtype=dtype) if isinstance(model, nd.DeLaN) else _CastAnalytical(model, dtype)
    if isinstance(model, nd.MLP):
        return FastMLP(model, dtype)
    raise TypeError(type(model))


class _CastAnalytical:
    def __init__(self, m: nd.Analytical, dtype):
        p = m.p
        self.Mt, self.mpl, self.J, self.mgl = (dtype(v) for v in (p.M_tot, p.mpl, p.J, p.mgl))
        self.bx, self.bth = dtype(p.b_x), dtype(p.b_th)
        self.dtype, self.name = dtype, m.name

    def qdd(self, q, qd, tau):
        th, thd = q[..., 1], qd[..., 1]
        c, s = np.cos(th), np.sin(th)
        r0 = tau[..., 0] + self.mpl * s * thd * thd - self.bx * qd[..., 0]
        r1 = tau[..., 1] + self.mgl * s - self.bth * thd
        M12 = self.mpl * c
        det = self.Mt * self.J - M12 * M12
        out = np.empty(q.shape, self.dtype)
        out[..., 0] = (self.J * r0 - M12 * r1) / det
        out[..., 1] = (self.Mt * r1 - M12 * r0) / det
        return out


# ---------------------------------------------------------------------------
# LQR from the model's own Jacobian
# ---------------------------------------------------------------------------
def lqr_from_model(model, gear: float, bryson=(1.0, 0.3, 2.0, 2.0), r: float = 1.0, eps: float = 1e-6):
    """Continuous-time LQR about upright using central-difference Jacobians of model.qdd (float64)."""
    from scipy.linalg import solve_continuous_are

    def f(s, u):
        q, qd = s[None, :2], s[None, 2:]
        tau = np.array([[gear * u, 0.0]])
        return np.concatenate([qd[0], model.qdd(q, qd, tau)[0]])

    s0 = np.zeros(4)
    A = np.stack([(f(s0 + eps * e, 0.0) - f(s0 - eps * e, 0.0)) / (2 * eps) for e in np.eye(4)], 1)
    B = ((f(s0, eps) - f(s0, -eps)) / (2 * eps))[:, None]
    Q = np.diag(1.0 / np.asarray(bryson) ** 2)
    P = solve_continuous_are(A, B, Q, r * np.eye(1))
    K = (B.T @ P / r).ravel()
    return K, np.linalg.eigvals(A - B @ K[None])


# ---------------------------------------------------------------------------
# MPPI
# ---------------------------------------------------------------------------
@dataclass
class MPPIConfig:
    K: int = 512
    H: int = 75
    dt_plan: float = 0.02
    sigma: float = 0.5
    ar: float = 0.8
    ess_lo: float = 0.1
    ess_hi: float = 0.2
    w_theta: float = 10.0
    w_x: float = 1.0
    x_scale: float = 1.0
    xd_scale: float = 2.0
    thd_scale: float = 5.0
    w_u: float = 0.1
    barrier: float = 1e3
    x_soft: float = 1.6
    terminal: float = 10.0
    w_energy: float = 0.0
    gamma_ctrl: float = 0.0     # weight of the IT control-cost term lam*gamma*sum u sigma^-2 eps (ISSUES C2)
    lqr: bool = True
    lqr_in: tuple = (0.3, 2.0)      # |th| < 0.3 and |thd| < 2 -> LQR
    lqr_out: tuple = (0.5, 4.0)     # hysteresis: back to MPPI if |th| > 0.5 or |thd| > 4
    gear: float = 10.0
    mgl: float = 0.4905
    J: float = 0.0344245927672956

    def as_dict(self):
        return asdict(self)


class MPPI:
    def __init__(self, model, cfg: MPPIConfig, seed: int = 0, dtype=np.float32):
        self.m, self.cfg, self.dtype = model, cfg, dtype
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros(cfg.H, dtype)
        self.lam = 1.0
        self.last_ess = None

    def stage_cost(self, q, qd, u):
        c = self.cfg
        x, th, xd, thd = q[..., 0], q[..., 1], qd[..., 0], qd[..., 1]
        cost = (c.w_theta * (1 - np.cos(th)) + c.w_x * (x / c.x_scale) ** 2 + (xd / c.xd_scale) ** 2
                + (thd / c.thd_scale) ** 2 + c.w_u * u * u
                + c.barrier * np.maximum(np.abs(x) - c.x_soft, 0) ** 2)
        if c.w_energy:
            E = 0.5 * c.J * thd * thd + c.mgl * np.cos(th)
            cost = cost + c.w_energy * ((E - c.mgl) / c.mgl) ** 2
        return cost

    def noise(self):
        c = self.cfg
        n = self.rng.standard_normal((c.H, c.K)).astype(self.dtype)
        eps = np.empty_like(n)
        a, s = self.dtype(c.ar), self.dtype(np.sqrt(1 - c.ar ** 2))
        eps[0] = n[0]
        for t in range(1, c.H):
            eps[t] = a * eps[t - 1] + s * n[t]
        return self.dtype(c.sigma) * eps

    def plan(self, q, qd):
        c, dt = self.cfg, self.dtype(self.cfg.dt_plan)
        eps = self.noise()
        Uk = np.clip(self.U[:, None] + eps, -1, 1)
        eps = Uk - self.U[:, None]                      # effective (clipped) perturbation
        Q = np.broadcast_to(q.astype(self.dtype), (c.K, 2)).copy()
        Qd = np.broadcast_to(qd.astype(self.dtype), (c.K, 2)).copy()
        S = np.zeros(c.K, self.dtype)
        tau = np.zeros((c.K, 2), self.dtype)
        f = self.m.qdd
        for t in range(c.H):
            tau[:, 0] = c.gear * Uk[t]
            S += self.stage_cost(Q, Qd, Uk[t])
            k1v = f(Q, Qd, tau)
            k2q = Qd + 0.5 * dt * k1v; k2v = f(Q + 0.5 * dt * Qd, k2q, tau)
            k3q = Qd + 0.5 * dt * k2v; k3v = f(Q + 0.5 * dt * k2q, k3q, tau)
            k4q = Qd + dt * k3v;       k4v = f(Q + dt * k3q, k4q, tau)
            Q = Q + dt / 6 * (Qd + 2 * k2q + 2 * k3q + k4q)
            Qd = Qd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        S += c.terminal * self.stage_cost(Q, Qd, np.zeros(c.K, self.dtype))
        S = np.where(np.isfinite(S), S, np.inf).astype(np.float64)
        ctrl = c.gamma_ctrl * (self.U[:, None] * eps).sum(0).astype(np.float64) / c.sigma ** 2
        w = self._weights(S, ctrl)
        self.U = (self.U + (eps * w[None, :].astype(self.dtype)).sum(1)).clip(-1, 1).astype(self.dtype)
        return float(self.U[0])

    def _weights(self, S, ctrl):
        """softmax(-(S + lam*ctrl)/lam), ctrl = gamma sum_t u_t sigma^-2 eps_t; lam by bisection on ESS."""
        c = self.cfg
        lo_t, hi_t = c.ess_lo * c.K, c.ess_hi * c.K
        target = 0.5 * (lo_t + hi_t)
        finite = np.isfinite(S)
        if not finite.any():
            return np.full(c.K, 1.0 / c.K)
        Smin = S[finite].min()

        def w_of(lam):
            z = -(S - Smin) / lam - ctrl
            z = np.where(finite, z, -np.inf)
            z -= z.max()
            w = np.exp(z)
            return w / w.sum()

        a, b = np.log(1e-6), np.log(1e6)
        lam = self.lam
        w = w_of(lam)
        ess = 1.0 / (w * w).sum()
        if not (lo_t <= ess <= hi_t):
            for _ in range(40):                       # ESS increases with lam
                m = 0.5 * (a + b)
                w = w_of(np.exp(m))
                ess = 1.0 / (w * w).sum()
                if ess < target:
                    a = m
                else:
                    b = m
                if lo_t <= ess <= hi_t:
                    break
            lam = float(np.exp(m))
        self.lam, self.last_ess = lam, float(ess)
        return w

    def shift(self):
        self.U = np.append(self.U[1:], self.U[-1]).astype(self.dtype)


# ---------------------------------------------------------------------------
# closed-loop episode on the analytical (= MuJoCo off-rail) plant
# ---------------------------------------------------------------------------
def dmc_initial_state(seed: int):
    """Exactly dm_control cartpole/swingup's reset for task_kwargs={'random': seed}."""
    r = np.random.RandomState(seed)
    x = 0.01 * r.randn()
    th = np.pi + 0.01 * r.randn()
    v = 0.01 * r.randn(2)
    return np.array([x, th]), v


def _tol_gauss(d, margin, vam=0.1):
    s = np.sqrt(-2 * np.log(vam))
    return np.exp(-0.5 * (d / margin * s) ** 2)


def dmc_reward(q, qd, u):
    """Re-implementation of dm_control cartpole swing-up (smooth) reward."""
    upright = (np.cos(q[1]) + 1) / 2
    centered = (1 + _tol_gauss(q[0], 2.0)) / 2
    small_control = (4 + max(0.0, 1 - u * u)) / 5
    small_velocity = (1 + _tol_gauss(qd[1], 5.0)) / 2
    return upright * small_control * small_velocity * centered


def balanced(q, qd):
    return abs(nd.wrap(q[1])) < 0.2 and abs(qd[1]) < 1.0


def run_episode(model, cfg: MPPIConfig, seed: int, plant=None, T_sim: float = 10.0,
                hold_s: float = 2.0, record: bool = False):
    p = nd.load_params()
    plant = plant or nd.Analytical(p)
    pm = planning_model(model)
    ctrl = MPPI(pm, cfg, seed=seed)
    K_lqr = lqr_from_model(model, p.gear)[0] if cfg.lqr else None
    q, qd = dmc_initial_state(seed)
    n_sim = int(round(T_sim / p.dt))
    sub = int(round(cfg.dt_plan / p.dt))
    in_lqr, switches, ret, max_x = False, 0, 0.0, 0.0
    bal = np.zeros(n_sim, bool)
    rail = False
    traj = dict(t=[], x=[], th=[], u=[], mode=[]) if record else None
    t_plan = []
    u = 0.0
    for k in range(n_sim):
        if k % sub == 0:
            th_w = nd.wrap(q[1])
            if cfg.lqr:
                if not in_lqr and abs(th_w) < cfg.lqr_in[0] and abs(qd[1]) < cfg.lqr_in[1]:
                    in_lqr, switches = True, switches + 1
                elif in_lqr and (abs(th_w) > cfg.lqr_out[0] or abs(qd[1]) > cfg.lqr_out[1]):
                    in_lqr, switches = False, switches + 1
            t0 = time.perf_counter()
            if in_lqr:
                s = np.array([q[0], th_w, qd[0], qd[1]])
                u = float(np.clip(-K_lqr @ s, -1, 1))
                ctrl.shift()                         # keep the MPPI plan warm
                ctrl.U[:] = np.clip(ctrl.U, -1, 1)
            else:
                u = ctrl.plan(q, qd)
                ctrl.shift()
            t_plan.append(time.perf_counter() - t0)
        ret += dmc_reward(q, qd, u)
        bal[k] = balanced(q, qd)
        if record:
            traj["t"].append(k * p.dt); traj["x"].append(q[0]); traj["th"].append(nd.wrap(q[1]))
            traj["u"].append(u); traj["mode"].append(int(in_lqr))
        q2, qd2 = nd.rk4_step(plant, q[None], qd[None], np.array([[p.gear * u, 0.0]]), p.dt)
        q, qd = q2[0], qd2[0]
        max_x = max(max_x, abs(q[0]))
        if abs(q[0]) >= p.x_limit:
            rail = True
            break
    # time-to-balance: first t at which 'balanced' then holds for hold_s
    hold = int(round(hold_s / p.dt))
    ttb = None
    run = 0
    for k in range(len(bal)):
        run = run + 1 if bal[k] else 0
        if run >= hold:
            ttb = (k - hold + 1) * p.dt
            break
    out = dict(seed=seed, success=bool(ttb is not None and not rail), time_to_balance=ttb, rail=rail,
               max_abs_x=float(max_x), dmc_return=float(ret), switches=switches,
               ms_per_plan=float(1e3 * np.mean(t_plan)) if t_plan else None,
               frac_balanced_last2s=float(bal[-hold:].mean()) if not rail else 0.0,
               balanced_curve=bal.astype(np.uint8))
    if record:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))
