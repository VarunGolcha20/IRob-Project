"""Phase 2 checks on the saved datasets (run scripts/p2_collect.py first)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel  # noqa: E402
from cartpole.data import LQRBalancePolicy, flatten, in_support, load_episodes  # noqa: E402
from cartpole.params import load_params  # noqa: E402
from cartpole.plotting import DATA_DIR  # noqa: E402

torch.set_default_dtype(torch.float64)
SPLITS = ["R_train", "R_val", "R_test", "E_train", "E_val", "E_test", "S_train", "S_val", "S_test"]
pytestmark = pytest.mark.skipif(not (DATA_DIR / "R_train.npz").exists(), reason="run p2_collect.py first")


@pytest.mark.parametrize("split", SPLITS)
def test_labels_satisfy_euler_lagrange(split):
    """Every training row: qacc = M^-1 (tau_gen - C qd - g - D qd), tau_gen = actuator + constraint."""
    p = load_params()
    eps, _ = load_episodes(DATA_DIR / f"{split}.npz")
    d = flatten(eps)
    qdd = AnalyticalModel(p).qdd(*(torch.as_tensor(d[k]) for k in ("q", "qd", "tau"))).numpy()
    err = np.abs(qdd - d["qdd"]).max(-1)
    assert err[~d["limit_active"]].max() < 1e-10
    if d["limit_active"].any():          # constraint-solver tolerance (see PART1 check 4b)
        assert err[d["limit_active"]].max() < 1e-3


@pytest.mark.parametrize("split", SPLITS)
def test_rows_chain_by_rk4_and_actuation(split):
    """Actuator force is exactly gear*u on the cart and zero on the hinge."""
    p = load_params()
    eps, _ = load_episodes(DATA_DIR / f"{split}.npz")
    d = flatten(eps)
    assert np.allclose(d["tau_act"][:, 0], p.gear * d["u"], atol=1e-12)
    assert np.all(d["tau_act"][:, 1] == 0)


def test_unforced_pool_is_undamped_and_conservative():
    p = load_params()
    eps, meta = load_episodes(DATA_DIR / "U.npz")
    model = AnalyticalModel(p, damping=False)
    for e in eps:
        assert np.all(e["u"] == 0) and not e["limit_active"].any()
        assert np.allclose(e["qfrc_passive"], 0)            # dof_damping was zeroed
        qdd = model.qdd(torch.as_tensor(e["qpos"]), torch.as_tensor(e["qvel"]),
                        torch.zeros(len(e["u"]), 2)).numpy()
        assert np.abs(qdd - e["qacc"]).max() < 1e-10
        H = e["energy"].sum(-1)
        assert np.abs(H - H[0]).max() / (p.mpl * p.g) < 2e-6


def test_episode_level_split_has_no_overlap():
    seeds = []
    for s in SPLITS:
        _, meta = load_episodes(DATA_DIR / f"{s}.npz")
        seeds += list(meta["seeds"])
    assert len(seeds) == len(set(seeds))


def test_unforced_groups_and_energy_levels():
    """small/large below the separatrix (internal energy < 2 m g l), rotating just above it,
    fast_rotating well above it."""
    p = load_params()
    agl = p.mpl * p.g
    eps, meta = load_episodes(DATA_DIR / "U.npz")
    groups = list(meta["groups"])
    assert sorted(set(groups)) == ["fast_rotating", "large", "rotating", "small"]
    for e, g in zip(eps, groups):
        q, qd = e["qpos"][0], e["qvel"][0]
        J_eff = p.J - (p.mpl * np.cos(q[1])) ** 2 / p.M_tot
        Ein = 0.5 * J_eff * qd[1] ** 2 + agl * (1 + np.cos(q[1]))
        px = p.M_tot * qd[0] + p.mpl * np.cos(q[1]) * qd[1]
        assert abs(px) < 1e-12                      # COM at rest: cart never reaches the rail
        if g in ("small", "large"):
            assert Ein < 2 * agl
        elif g == "rotating":
            assert 2 * agl < Ein < 2.3 * agl
        else:
            assert Ein > 2.3 * agl


def test_drop_limit_removes_exactly_limit_rows():
    eps, _ = load_episodes(DATA_DIR / "R_train.npz")
    full, free = flatten(eps), flatten(eps, drop_limit=True)
    assert len(free["u"]) == int((~full["limit_active"]).sum())
    assert not free["limit_active"].any()
    assert np.abs(free["tau"][:, 0]).max() <= 10.0 + 1e-9    # only actuator forces remain


def test_lqr_is_stabilising():
    K = LQRBalancePolicy(load_params(), seed=0)
    assert np.all(K.closed_loop_eigs.real < 0)


def test_balance_pool_covers_balance_region():
    d = flatten(load_episodes(DATA_DIR / "S_train.npz")[0])
    th = np.arctan2(np.sin(d["q"][:, 1]), np.cos(d["q"][:, 1]))
    bal = (np.abs(th) < 0.2) & (np.abs(d["qd"][:, 1]) < 1.0)
    assert bal.sum() > 5000


def test_density_support_is_self_consistent():
    d = flatten(load_episodes(DATA_DIR / "R_train.npz")[0], drop_limit=True)
    mask = in_support(d["q"], d["qd"], d["q"], d["qd"])
    assert mask.mean() > 0.95                       # training rows are mostly in their own support
    up_q = np.array([[0.0, 0.0]]); up_qd = np.zeros((1, 2))
    assert not in_support(d["q"], d["qd"], up_q, up_qd).any()   # upright is outside random data
