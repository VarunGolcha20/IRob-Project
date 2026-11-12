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


class EnergyPumpingPolicy:
    """Astrom-Furuta energy pumping with cart centering and OU exploration noise.

    Pole energy in the cart frame: E_p = 1/2 J thd^2 + m_p g l cos(th).
    The pole equation gives dE_p/dt = -m_p l cos(th) thd xdd, so commanding
        xdd = k_E (E_p - E_ref)/(m_p g l) * thd cos(th)
    makes d/dt [1/2 (E_p - E_ref)^2] = -k_E m_p l (E_p - E_ref)^2 (thd cos th)^2 / (m_p g l) <= 0.
    E_ref is redrawn every `resample_s` seconds, so one episode visits several
    energy levels (small swings, near-separatrix, full rotations).
    Uses nominal parameters: it only collects data, it is never evaluated.
    """

    def __init__(self, p, seed: int, k_E: float = 6.0, k_x: float = 3.0, k_v: float = 3.0,
                 sigma: float = 0.3, tau_c: float = 0.2, E_range=(-0.5, 1.5),
                 resample_s: float = 2.5):
        self.p, self.k_E, self.k_x, self.k_v = p, k_E, k_x, k_v
        self.rng = np.random.default_rng(seed)
        self.E_range = E_range
        self.n_resample = int(round(resample_s / p.dt))
        self.alpha = np.exp(-p.dt / tau_c)                       # OU decay per step
        self.noise_scale = sigma * np.sqrt(1.0 - self.alpha ** 2)  # stationary std = sigma
        self.noise = 0.0
        self.E_ref = 0.0
        self.E_refs = []

    def __call__(self, physics, k: int) -> float:
        p = self.p
        agl = p.mpl * p.g
        if k % self.n_resample == 0:
            self.E_ref = self.rng.uniform(*self.E_range) * agl
            self.E_refs.append(self.E_ref / agl)
        x, th = physics.data.qpos
        xd, thd = physics.data.qvel
        E_p = 0.5 * p.J * thd ** 2 + agl * np.cos(th)
        xdd = self.k_E * (E_p - self.E_ref) / agl * thd * np.cos(th) - self.k_x * x - self.k_v * xd
        self.noise = self.alpha * self.noise + self.noise_scale * self.rng.standard_normal()
        return p.M_tot * xdd / p.gear + self.noise


class LQRBalancePolicy:
    """LQR balance about the upright, plus OU exploration noise (balance pool S).

    Linearising the Euler-Lagrange model at q = 0 (PART1 section 3), with
    s = (x, th, xd, thd) and th wrapped to (-pi, pi]:
        M0 qdd = [gear u, 0] + [0, m_p g l th] - D qd,   M0 = [[M_t, m_p l], [m_p l, J]]
        A = [[0, I], [M0^-1 G, -M0^-1 D]],  G = [[0, 0], [0, m_p g l]]
        B = [0; M0^-1 [gear, 0]^T]
    Weights by Bryson's rule, Q_ii = 1 / (max acceptable s_i)^2:
        |x| 1 m, |th| 0.3 rad, |xd| 2 m/s, |thd| 2 rad/s;  R = 1 (|u| <= 1).
    The noise amplitude sigma is drawn per episode, so some episodes hold the
    pole near upright and others lose it and fall, covering the whole
    neighbourhood of the balance point.
    """

    def __init__(self, p, seed: int, sigma_range=(0.5, 1.5), tau_c: float = 0.2,
                 bryson=(1.0, 0.3, 2.0, 2.0)):
        from scipy.linalg import solve_continuous_are
        self.p = p
        self.rng = np.random.default_rng(seed)
        M0 = np.array([[p.M_tot, p.mpl], [p.mpl, p.J]])
        Minv = np.linalg.inv(M0)
        G = np.array([[0.0, 0.0], [0.0, p.mpl * p.g]])
        D = np.diag([p.b_x, p.b_th])
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [Minv @ G, -Minv @ D]])
        B = np.concatenate([np.zeros(2), Minv @ np.array([p.gear, 0.0])])[:, None]
        Q = np.diag(1.0 / np.asarray(bryson) ** 2)
        R = np.eye(1)
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P).ravel()
        self.closed_loop_eigs = np.linalg.eigvals(A - B @ self.K[None, :])
        self.sigma = self.rng.uniform(*sigma_range)
        self.alpha = np.exp(-p.dt / tau_c)
        self.noise_scale = self.sigma * np.sqrt(1.0 - self.alpha ** 2)
        self.noise = 0.0

    def __call__(self, physics, k: int) -> float:
        x, th = physics.data.qpos
        xd, thd = physics.data.qvel
        s = np.array([x, np.arctan2(np.sin(th), np.cos(th)), xd, thd])
        self.noise = self.alpha * self.noise + self.noise_scale * self.rng.standard_normal()
        return float(-self.K @ s + self.noise)


def state_hook(qpos, qvel):
    """physics_hook: set the initial state (model unchanged)."""
    def hook(physics):
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
    return hook


def balance_initial_state(seed: int):
    """Near-upright start for the balance pool: |th| <= 0.3 rad, |thd| <= 1 rad/s,
    |x| <= 0.5 m, |xd| <= 0.5 m/s."""
    rng = np.random.default_rng(seed + 10_000)
    return (np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)]),
            np.array([rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0)]))


def unforced_initial_states(p, n_per_group: int = 10, seed: int = 600):
    """Initial states for the unforced, undamped energy-drift pool.

    Three groups of increasing energy. The cart velocity is set so that
    p_x = M_t xd + m_p l cos(th) thd = 0: the system's COM then stays put and
    the cart never reaches the rail limit.
      small:    phi = th - pi in [-0.5, 0.5] rad, thd in [-1, 1]  (inside random-data range)
      large:    released from rest with cos(th0) in [-0.5, 0.95] (60-162 deg swings)
      rotating: at the bottom with |thd0| in [7.9, 8.3] rad/s: just above the
                separatrix speed sqrt(4 m_p g l / J_eff) = 7.81 and inside the
                energy-pumping training support (max |thd| = 8.40)
      fast_rotating: at the bottom with |thd0| in [8.5, 11] rad/s: outside every
                training set, kept as an explicit extrapolation test
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_per_group):
        out.append(("small", np.pi + rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)))
    for i in range(n_per_group):
        th0 = np.arccos(rng.uniform(-0.5, 0.95)) * rng.choice([-1, 1])
        out.append(("large", th0, 0.0))
    for i in range(n_per_group):
        out.append(("rotating", np.pi, rng.uniform(7.9, 8.3) * rng.choice([-1, 1])))
    for i in range(n_per_group):
        out.append(("fast_rotating", np.pi, rng.uniform(8.5, 11.0) * rng.choice([-1, 1])))
    states = []
    for group, th0, thd0 in out:
        xd0 = -p.mpl * np.cos(th0) * thd0 / p.M_tot
        states.append(dict(group=group, qpos=np.array([0.0, th0]), qvel=np.array([xd0, thd0])))
    return states


def undamped_hook(qpos, qvel):
    """physics_hook: zero all damping (model.dof_damping[:] = 0) and set the state."""
    def hook(physics):
        physics.model.dof_damping[:] = 0.0
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
    return hook


def flatten(episodes: list[dict], drop_limit: bool = False) -> dict:
    """Stack episodes into training arrays.

    tau = generalized force = qfrc_actuator + qfrc_constraint (rail limit force is
    just another generalized force in the Euler-Lagrange equations).

    drop_limit=True removes rows where the rail limit is active. Training uses
    this (docs/ISSUES.md D2): limit rows carry impulsive constraint forces up to
    53 N, five times the actuator range, which no test-time use (rollouts away
    from the rail, MPPI with |x| < 1.6 m) ever sees. Both models are trained on
    the same rows, so the comparison stays fair.
    """
    out = {
        "q": np.concatenate([e["qpos"] for e in episodes]),
        "qd": np.concatenate([e["qvel"] for e in episodes]),
        "tau": np.concatenate([e["qfrc_actuator"] + e["qfrc_constraint"] for e in episodes]),
        "tau_act": np.concatenate([e["qfrc_actuator"] for e in episodes]),
        "u": np.concatenate([e["u"] for e in episodes]),
        "qdd": np.concatenate([e["qacc"] for e in episodes]),
        "limit_active": np.concatenate([e["limit_active"] for e in episodes]),
        "ep_id": np.concatenate([np.full(len(e["u"]), i) for i, e in enumerate(episodes)]),
        "t": np.concatenate([e["t"] for e in episodes]),
    }
    if drop_limit:
        keep = ~out["limit_active"]
        out = {k: v[keep] for k, v in out.items()}
    return out


# Density-based support of a training set in the (theta, theta_dot) plane
# (docs/ISSUES.md D1). A state counts as "in support" if its bin holds at
# least SUPPORT_MIN_COUNT training rows (>= 0.05 s of data). The dynamics do
# not depend on x (Noether, PART1 P2), and xd enters only through the damping,
# so (theta, theta_dot) is the plane in which the networks must generalise.
SUPPORT_BIN_TH = 0.1        # rad
SUPPORT_BIN_THD = 0.25      # rad/s
SUPPORT_THD_MAX = 12.0      # rad/s, histogram range
SUPPORT_MIN_COUNT = 5


def support_histogram(q: np.ndarray, qd: np.ndarray):
    th = np.arctan2(np.sin(q[:, 1]), np.cos(q[:, 1]))
    eth = np.arange(-np.pi, np.pi + 1e-9, SUPPORT_BIN_TH)
    ethd = np.arange(-SUPPORT_THD_MAX, SUPPORT_THD_MAX + 1e-9, SUPPORT_BIN_THD)
    H, _, _ = np.histogram2d(th, qd[:, 1], bins=[eth, ethd])
    return H, eth, ethd


def in_support(train_q, train_qd, q, qd) -> np.ndarray:
    """Boolean mask: rows (q, qd) whose (theta, theta_dot) bin has >= SUPPORT_MIN_COUNT training rows."""
    H, eth, ethd = support_histogram(train_q, train_qd)
    th = np.arctan2(np.sin(q[:, 1]), np.cos(q[:, 1]))
    i = np.clip(np.digitize(th, eth) - 1, 0, H.shape[0] - 1)
    j = np.digitize(qd[:, 1], ethd) - 1
    ok = (j >= 0) & (j < H.shape[1])
    out = np.zeros(len(th), bool)
    out[ok] = H[i[ok], j[ok]] >= SUPPORT_MIN_COUNT
    return out


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
