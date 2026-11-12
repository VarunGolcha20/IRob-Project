"""Phase 2 checks on the saved datasets (run scripts/p2_collect.py first)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel  # noqa: E402
from cartpole.data import flatten, load_episodes  # noqa: E402
from cartpole.params import load_params  # noqa: E402
from cartpole.plotting import DATA_DIR  # noqa: E402

torch.set_default_dtype(torch.float64)
SPLITS = ["R_train", "R_val", "R_test", "E_train", "E_val", "E_test"]
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
