"""Phase 1 checks: the analytical EL model is the MuJoCo cart-pole (pytest -q)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel  # noqa: E402
from cartpole.data import random_policy, run_episode  # noqa: E402
from cartpole.integrators import rollout  # noqa: E402
from cartpole.params import capsule_inertia_perp, load_params, make_env  # noqa: E402

torch.set_default_dtype(torch.float64)
T = torch.as_tensor


@pytest.fixture(scope="module")
def setup():
    env = make_env(seed=7)
    p = load_params(env.physics)
    ep = run_episode(env, random_policy(7), n_steps=400)
    return env, p, AnalyticalModel(p), ep


def test_capsule_inertia(setup):
    env, p, _, _ = setup
    m = env.physics.model
    r, half = m.geom_size[m.name2id("pole_1", "geom")][:2]
    assert capsule_inertia_perp(p.m_p, r, 2 * half) == pytest.approx(p.I_c, rel=1e-12)


def test_structure_matches_mujoco(setup):
    _, _, model, ep = setup
    q, qd = T(ep["qpos"]), T(ep["qvel"])
    assert torch.allclose(model.mass_matrix(q).reshape(-1, 4), T(ep["fullM"]).reshape(-1, 4), atol=1e-12)
    assert torch.allclose(model.bias(q, qd), T(ep["qfrc_bias"]), atol=1e-12)
    assert torch.allclose(-model.damping_force(qd), T(ep["qfrc_passive"]), atol=1e-14)
    assert torch.allclose(model.energy(q, qd, mujoco_offset=True), T(ep["energy"]).sum(-1), atol=1e-12)


def test_forward_dynamics_free_rows(setup):
    _, _, model, ep = setup
    free = ~ep["limit_active"]
    tau = T(ep["qfrc_actuator"] + ep["qfrc_constraint"])
    qdd = model.qdd(T(ep["qpos"]), T(ep["qvel"]), tau)
    assert torch.allclose(qdd[free], T(ep["qacc"])[free], atol=1e-11)


def test_inverse_is_left_inverse(setup):
    _, _, model, ep = setup
    q, qd, tau = T(ep["qpos"]), T(ep["qvel"]), T(ep["qfrc_actuator"])
    assert torch.allclose(model.inverse(q, qd, model.qdd(q, qd, tau)), tau, atol=1e-12)


def test_rk4_equals_mujoco():
    env = make_env(seed=3)
    p = load_params(env.physics)
    ep = run_episode(env, lambda ph, k: 0.3 * np.sin(3.9 * k * p.dt), n_steps=200)
    assert not ep["limit_active"].any()
    tau = torch.zeros(200, 2)
    tau[:, 0] = p.gear * T(ep["u"])
    qs, qds = rollout(AnalyticalModel(p), T(ep["qpos"][0]), T(ep["qvel"][0]), tau, p.dt)
    assert torch.allclose(qs[-1], T(ep["qpos_final"]), atol=1e-12)
    assert torch.allclose(qds[-1], T(ep["qvel_final"]), atol=1e-12)


def test_passivity_identity():
    """qd^T (Mdot - 2C) qd = 0  (energy balance dH/dt = qd^T (tau - D qd))."""
    p = load_params()
    model = AnalyticalModel(p)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(100, 2, generator=g) * 3
    qd = torch.randn(100, 2, generator=g) * 3
    dM_dth = -p.mpl * torch.sin(q[:, 1])
    Mdot = torch.zeros(100, 2, 2)
    Mdot[:, 0, 1] = Mdot[:, 1, 0] = dM_dth * qd[:, 1]
    Mdot_qd = (Mdot @ qd.unsqueeze(-1)).squeeze(-1)
    val = (qd * (Mdot_qd - 2 * model.coriolis(q, qd))).sum(-1)
    assert val.abs().max() < 1e-12


def test_energy_conserved_without_damping():
    """Unforced, undamped: H constant up to RK4's O(h^4) error, p_x exactly conserved
    in continuous time (checked to the same order)."""
    p = load_params()
    model = AnalyticalModel(p, damping=False)
    q0, qd0 = T([0.0, 3.0]), T([0.5, 2.0])
    qs, qds = rollout(model, q0, qd0, torch.zeros(1000, 2), p.dt)
    H = model.energy(qs, qds)
    px = model.momentum_x(qs, qds)
    assert (H - H[0]).abs().max() / (p.mpl * p.g) < 1e-6
    assert (px - px[0]).abs().max() < 1e-6
