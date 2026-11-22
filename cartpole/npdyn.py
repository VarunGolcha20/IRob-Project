"""Torch-free, MuJoCo-free float64 dynamics: checkpoint loader, analytical model,
MLP, DeLaN, batched RK4.

Why this exists
---------------
Evaluation (Phase 4) and MPPI (Phase 5) only ever run the trained networks
forward. This module does that in pure numpy, in float64 (PLAN 3.4: "all
evaluations run in float64"), with no torch or MuJoCo import. It:
  * reads `weights/*.pt` directly (a torch checkpoint is a zip of a pickle plus
    raw little-endian storages), so it works on machines without torch;
  * reproduces `cartpole.models.{MLPModel, DeLaN}.qdd` exactly (checked against
    the stored best validation losses, see tests/test_npdyn.py);
  * is batched over any leading shape, which is what MPPI needs (K x 2 states).

Every model exposes the common interface ``qdd(q, qd, tau) -> qdd`` with
arrays ending in 2, like the torch models.
"""
from __future__ import annotations

import json
import math
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weights"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# torch checkpoint reader (no torch needed)
# ---------------------------------------------------------------------------
_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *_, **__):
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()


def _rebuild_parameter(data, *_, **__):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf: zipfile.ZipFile, prefix: str):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module.startswith("torch"):
            raise pickle.UnpicklingError(f"unsupported torch object {module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, stype, key, _loc, _numel = pid
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            self.cache[key] = np.frombuffer(raw, dtype=np.dtype(stype.dtype).newbyteorder("<"))
        return self.cache[key]


def load_checkpoint(path) -> dict:
    """Load a torch.save()'d dict with every tensor as a numpy array."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        import io
        return _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def softplus(z):
    # torch.nn.functional.softplus with threshold 20 (linear above it)
    return np.where(z > 20.0, z, np.log1p(np.exp(np.minimum(z, 20.0))))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def solve2(M11, M12, M22, b0, b1):
    det = M11 * M22 - M12 * M12
    return np.stack([(M22 * b0 - M12 * b1) / det, (M11 * b1 - M12 * b0) / det], -1)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# analytical model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    m_c: float = 1.0
    m_p: float = 0.1
    l: float = 0.5
    I_c: float = 0.009424592767295598
    b_x: float = 5e-4
    b_th: float = 2e-6
    gear: float = 10.0
    g: float = 9.81
    dt: float = 0.01
    x_limit: float = 1.8

    @property
    def M_tot(self):
        return self.m_c + self.m_p

    @property
    def mpl(self):
        return self.m_p * self.l

    @property
    def J(self):
        return self.I_c + self.m_p * self.l ** 2

    @property
    def mgl(self):
        return self.m_p * self.g * self.l


def load_params() -> Params:
    """Parameters as read from MuJoCo in Phase 1 (results/p1_validation.json)."""
    p = json.loads((RES_DIR / "p1_validation.json").read_text())["params"]
    return Params(**{k: p[k] for k in Params.__dataclass_fields__})


class Analytical:
    name = "analytical"

    def __init__(self, p: Params, damping: bool = True):
        self.p, self.damping = p, damping

    def structure(self, th):
        p, c, s = self.p, np.cos(th), np.sin(th)
        one = np.ones_like(th)
        return dict(M11=p.M_tot * one, M12=p.mpl * c, M22=p.J * one,
                    dM11=0 * one, dM12=-p.mpl * s, dM22=0 * one,
                    V=p.mgl * c, dV=-p.mgl * s)

    def qdd(self, q, qd, tau):
        p = self.p
        th, thd, xd = q[..., 1], qd[..., 1], qd[..., 0]
        c, s = np.cos(th), np.sin(th)
        bx, bth = (p.b_x, p.b_th) if self.damping else (0.0, 0.0)
        r0 = tau[..., 0] + p.mpl * s * thd * thd - bx * xd
        r1 = tau[..., 1] + p.mgl * s - bth * thd
        return solve2(p.M_tot, p.mpl * c, p.J, r0, r1)

    def energy(self, q, qd):
        return true_energy(self.p, q, qd)


def true_energy(p: Params, q, qd):
    """H_true = T + V with V = m_p g l cos(th) (upright = +mgl, hanging = -mgl)."""
    c = np.cos(q[..., 1])
    xd, thd = qd[..., 0], qd[..., 1]
    return 0.5 * (p.M_tot * xd ** 2 + 2 * p.mpl * c * xd * thd + p.J * thd ** 2) + p.mgl * c


# ---------------------------------------------------------------------------
# learned models
# ---------------------------------------------------------------------------
class MLP:
    name = "mlp"

    def __init__(self, ck: dict):
        sd, st = ck["state_dict"], ck["stats"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net.")})
        self.W = [sd[f"net.{i}.weight"].astype(np.float64).T.copy() for i in idx]
        self.b = [sd[f"net.{i}.bias"].astype(np.float64) for i in idx]
        f = lambda k: np.asarray(sd.get(k, st[k]), np.float64)  # noqa: E731
        self.in_mean, self.in_std = f("in_mean"), f("in_std")
        self.out_mean, self.out_std = f("qdd_mean"), f("qdd_std")
        self.ck = ck

    def qdd(self, q, qd, tau):
        th = q[..., 1]
        z = np.stack([q[..., 0], np.sin(th), np.cos(th), qd[..., 0], qd[..., 1], tau[..., 0]], -1)
        h = (z - self.in_mean) / self.in_std
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = softplus(h @ W + b)
        return (h @ self.W[-1] + self.b[-1]) * self.out_std + self.out_mean


class _DualMLP:
    def __init__(self, sd, prefix):
        idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith(prefix + ".hidden.")})
        self.W = [sd[f"{prefix}.hidden.{i}.weight"].astype(np.float64).T.copy() for i in idx]
        self.b = [sd[f"{prefix}.hidden.{i}.bias"].astype(np.float64) for i in idx]
        self.Wo = sd[f"{prefix}.out.weight"].astype(np.float64).T.copy()
        self.bo = sd[f"{prefix}.out.bias"].astype(np.float64)

    def __call__(self, th):
        s, c = np.sin(th), np.cos(th)
        h, dh = np.stack([s, c], -1), np.stack([c, -s], -1)
        for W, b in zip(self.W, self.b):
            z = h @ W + b
            h, dh = softplus(z), sigmoid(z) * (dh @ W)
        return h @ self.Wo + self.bo, dh @ self.Wo


class DeLaN:
    name = "delan"

    def __init__(self, ck: dict):
        sd = ck["state_dict"]
        self.eps = float(ck["model_config"]["eps"])
        self.L, self.Vn = _DualMLP(sd, "L_net"), _DualMLP(sd, "V_net")
        self.d = softplus(sd["d_raw"].astype(np.float64))
        self.damping_on = True
        self.ck = ck

    def structure(self, th):
        l, dl = self.L(th)
        V, dV = self.Vn(th)
        r11, l21, r22 = l[..., 0], l[..., 1], l[..., 2]
        dr11, dl21, dr22 = dl[..., 0], dl[..., 1], dl[..., 2]
        l11, l22 = softplus(r11), softplus(r22)
        dl11, dl22 = sigmoid(r11) * dr11, sigmoid(r22) * dr22
        return dict(M11=l11 ** 2 + self.eps, M12=l11 * l21, M22=l21 ** 2 + l22 ** 2 + self.eps,
                    dM11=2 * l11 * dl11, dM12=dl11 * l21 + l11 * dl21,
                    dM22=2 * l21 * dl21 + 2 * l22 * dl22, V=V[..., 0], dV=dV[..., 0])

    def _terms(self, q, qd, s):
        xd, thd = qd[..., 0], qd[..., 1]
        dMqd0 = s["dM11"] * xd + s["dM12"] * thd
        dMqd1 = s["dM12"] * xd + s["dM22"] * thd
        half = 0.5 * (s["dM11"] * xd ** 2 + 2 * s["dM12"] * xd * thd + s["dM22"] * thd ** 2)
        d = self.d if self.damping_on else np.zeros(2)
        c0 = thd * dMqd0 + d[0] * xd
        c1 = thd * dMqd1 - half + s["dV"] + d[1] * thd
        return c0, c1

    def qdd(self, q, qd, tau):
        s = self.structure(q[..., 1])
        c0, c1 = self._terms(q, qd, s)
        return solve2(s["M11"], s["M12"], s["M22"], tau[..., 0] - c0, tau[..., 1] - c1)

    def inverse(self, q, qd, qdd):
        s = self.structure(q[..., 1])
        c0, c1 = self._terms(q, qd, s)
        a0, a1 = qdd[..., 0], qdd[..., 1]
        return np.stack([s["M11"] * a0 + s["M12"] * a1 + c0, s["M12"] * a0 + s["M22"] * a1 + c1], -1), s

    def energy(self, q, qd):
        s = self.structure(q[..., 1])
        xd, thd = qd[..., 0], qd[..., 1]
        return 0.5 * (s["M11"] * xd ** 2 + 2 * s["M12"] * xd * thd + s["M22"] * thd ** 2) + s["V"]


def load_model(name_or_path):
    """Load `weights/<name>.pt` (or a path) as a numpy model."""
    path = Path(name_or_path)
    if not path.suffix:
        path = WEIGHT_DIR / f"{name_or_path}.pt"
    ck = load_checkpoint(path)
    return (MLP if ck["kind"] == "mlp" else DeLaN)(ck)


def run_name(kind, regime, N, seed, lam_inv=1.0):
    n = f"{kind}_{regime}{N}_s{seed}"
    return n + (f"_inv{lam_inv:g}" if kind == "delan" and lam_inv > 0 else "")


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------
def rk4_step(model, q, qd, tau, h):
    k1q, k1v = qd, model.qdd(q, qd, tau)
    k2q = qd + 0.5 * h * k1v; k2v = model.qdd(q + 0.5 * h * k1q, k2q, tau)
    k3q = qd + 0.5 * h * k2v; k3v = model.qdd(q + 0.5 * h * k2q, k3q, tau)
    k4q = qd + h * k3v;       k4v = model.qdd(q + h * k3q, k4q, tau)
    return (q + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q),
            qd + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))


def rollout(model, q0, qd0, tau_seq, h):
    """tau_seq: (T, ..., 2). Returns q, qd of shape (T+1, ..., 2)."""
    T = tau_seq.shape[0]
    qs = np.empty((T + 1,) + q0.shape); qds = np.empty_like(qs)
    qs[0], qds[0] = q0, qd0
    q, qd = q0, qd0
    for t in range(T):
        q, qd = rk4_step(model, q, qd, tau_seq[t], h)
        qs[t + 1], qds[t + 1] = q, qd
    return qs, qds


# ---------------------------------------------------------------------------
# data (same layout as cartpole.data, numpy only)
# ---------------------------------------------------------------------------
def load_episodes(split: str):
    z = np.load(DATA_DIR / f"{split}.npz")
    n = int(z["n_episodes"])
    eps = [dict() for _ in range(n)]
    for key in z.files:
        if key.startswith("ep"):
            i, field = key.split("/", 1)
            eps[int(i[2:])][field] = z[key]
    return eps


def flatten(episodes, drop_limit=True):
    out = {
        "q": np.concatenate([e["qpos"] for e in episodes]),
        "qd": np.concatenate([e["qvel"] for e in episodes]),
        "tau": np.concatenate([e["qfrc_actuator"] + e["qfrc_constraint"] for e in episodes]),
        "qdd": np.concatenate([e["qacc"] for e in episodes]),
        "limit_active": np.concatenate([e["limit_active"] for e in episodes]),
        "ep_id": np.concatenate([np.full(len(e["u"]), i) for i, e in enumerate(episodes)]),
    }
    if drop_limit:
        keep = ~out["limit_active"]
        out = {k: v[keep] for k, v in out.items()}
    return out


def train_set(ck: dict):
    """Rebuild the exact training rows of a checkpoint from its stored episode choice."""
    eps = []
    for split, idx in ck["chosen"].items():
        pool = load_episodes(split)
        eps += [pool[i] for i in idx]
    return flatten(eps, drop_limit=True)


def val_set(regime: str):
    splits = ["R_val"] + (["E_val", "S_val"] if regime == "B" else [])
    return flatten(sum((load_episodes(s) for s in splits), []), drop_limit=True)


def forward_loss(model, data, qdd_std):
    pred = model.qdd(data["q"], data["qd"], data["tau"])
    return float((((pred - data["qdd"]) / qdd_std) ** 2).sum(-1).mean())
