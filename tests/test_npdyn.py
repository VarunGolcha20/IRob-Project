"""Torch-free backend (cartpole.npdyn) and MPPI building blocks (cartpole.mppi).

These tests need only numpy/scipy, so they also run on machines without torch or MuJoCo.
"""
import json

import numpy as np
import pytest

from cartpole import mppi as mp
from cartpole import npdyn as nd

P = nd.load_params()


@pytest.fixture(scope="module")
def test_rows():
    return nd.flatten(sum((nd.load_episodes(s) for s in ("R_test", "E_test", "S_test")), []))


def test_checkpoint_reproduces_stored_val_loss():
    """The numpy port of every grid model reproduces torch's best validation loss (float32 vs float64 gap only)."""
    info = json.loads((nd.RES_DIR / "p3_training.json").read_text())
    vals = {}
    for r in info["runs"]:
        m = nd.load_model(r["name"])
        reg = r["regime"]
        vals.setdefault(reg, nd.val_set(reg))
        v = nd.forward_loss(m, vals[reg], np.asarray(m.ck["stats"]["qdd_std"], float))
        assert abs(v / r["best_val"] - 1) < 1e-3, r["name"]


def test_analytical_matches_mujoco_labels(test_rows):
    d = test_rows
    res = np.abs(nd.Analytical(P).qdd(d["q"], d["qd"], d["tau"]) - d["qdd"]).max()
    assert res < 1e-12


def test_rk4_matches_mujoco_next_state():
    for e in nd.load_episodes("E_test")[:3]:
        q, qd = nd.rk4_step(nd.Analytical(P), e["qpos"][:-1], e["qvel"][:-1], e["qfrc_actuator"][:-1], P.dt)
        ok = ~(e["limit_active"][:-1] | e["limit_active"][1:])
        assert np.abs(q - e["qpos"][1:])[ok].max() < 1e-12
        assert np.abs(qd - e["qvel"][1:])[ok].max() < 1e-12


def test_delan_inverse_forward_identity(test_rows):
    m = nd.load_model("delan_B20_s0_inv1")
    d = test_rows
    qdd = m.qdd(d["q"], d["qd"], d["tau"])
    tau_hat, _ = m.inverse(d["q"], d["qd"], qdd)
    assert np.abs(tau_hat - d["tau"]).max() < 1e-9


def test_theta_table_matches_delan(test_rows):
    """Lookup-table DeLaN used by MPPI is the same function to << model error."""
    d = test_rows
    m = nd.load_model("delan_B20_s0_inv1")
    t = mp.ThetaTable(m, dtype=np.float64)
    err = np.abs(t.qdd(d["q"], d["qd"], d["tau"]) - m.qdd(d["q"], d["qd"], d["tau"])) / d["qdd"].std(0)
    assert err.max() < 1e-6


def test_fast_mlp_matches_mlp(test_rows):
    d = test_rows
    m = nd.load_model("mlp_B20_s0")
    f = mp.FastMLP(m, dtype=np.float64)
    assert np.abs(f.qdd(d["q"], d["qd"], d["tau"]) - m.qdd(d["q"], d["qd"], d["tau"])).max() < 1e-9


def test_dmc_initial_state_replica():
    """mppi.dmc_initial_state(s) == dm_control swingup reset with task_kwargs={'random': s}."""
    z = np.load(nd.DATA_DIR / "R_test.npz")
    for s, e in zip(z["meta/seeds"], nd.load_episodes("R_test")):
        q, v = mp.dmc_initial_state(int(s))
        assert np.allclose(q, e["qpos"][0], atol=1e-15) and np.allclose(v, e["qvel"][0], atol=1e-15)


def test_lqr_from_model_matches_pool_S_gain():
    K, eig = mp.lqr_from_model(nd.Analytical(P), P.gear)
    K_ref = json.loads((nd.RES_DIR / "p2_dataset.json").read_text())["lqr"]["K"]
    assert np.allclose(K, K_ref, rtol=1e-6)      # same u = -K s convention as LQRBalancePolicy
    assert (eig.real < 0).all()


def test_wilson():
    lo, hi = mp.wilson(18, 20)
    assert abs(lo - 0.699) < 2e-3 and abs(hi - 0.972) < 2e-3
