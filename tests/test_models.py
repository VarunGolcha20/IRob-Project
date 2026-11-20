"""Phase 3 structural tests for the learned models (PLAN 3.5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel  # noqa: E402
from cartpole.integrators import rollout  # noqa: E402
from cartpole.models import DeLaN, MLPModel  # noqa: E402
from cartpole.models.delan import lagrangian_dynamics  # noqa: E402
from cartpole.params import load_params  # noqa: E402

torch.set_default_dtype(torch.float64)


def random_delan(seed=0):
    torch.manual_seed(seed)
    return DeLaN().double()


def random_states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.stack([torch.rand(n, generator=g) * 3.6 - 1.8, torch.rand(n, generator=g) * 4 * np.pi - 2 * np.pi], -1)
    qd = torch.randn(n, 2, generator=g) * torch.tensor([1.0, 4.0])
    tau = torch.stack([torch.rand(n, generator=g) * 20 - 10, torch.zeros(n)], -1)
    return q, qd, tau


def dummy_stats():
    return dict(in_mean=torch.zeros(6), in_std=torch.ones(6), qdd_mean=torch.zeros(2), qdd_std=torch.ones(2))


def test_parameter_counts_are_matched():
    n_mlp = sum(p.numel() for p in MLPModel(dummy_stats()).parameters())
    n_delan = sum(p.numel() for p in DeLaN().parameters())
    assert (n_mlp, n_delan) == (34178, 34310)


def test_1_mass_matrix_is_spd_with_floor_eps():
    model = random_delan()
    th = torch.linspace(-2 * np.pi, 2 * np.pi, 10_000)
    s = model.structure(th)
    M = torch.stack([torch.stack([s["M11"], s["M12"]], -1), torch.stack([s["M12"], s["M22"]], -1)], -2)
    assert torch.equal(M, M.transpose(-1, -2))
    assert torch.linalg.eigvalsh(M).min() >= model.eps * (1 - 1e-9)


def test_2_passivity_and_energy_conservation_untrained():
    """qd^T (Mdot - 2C) qd = 0 for a random network; H conserved when tau = 0, D = 0."""
    model = random_delan(1)
    q, qd, _ = random_states(2000)
    s = model.structure(q[:, 1])
    thd = qd[:, 1]
    Mdot_qd = torch.stack([(s["dM11"] * qd[:, 0] + s["dM12"] * thd) * thd,
                           (s["dM12"] * qd[:, 0] + s["dM22"] * thd) * thd], -1)
    _, terms = model.forward_terms(q, qd, torch.zeros_like(qd))
    val = (qd * (Mdot_qd - 2 * terms["Cqd"])).sum(-1)
    scale = (qd * Mdot_qd).sum(-1).abs().max()
    assert val.abs().max() < 1e-12 * scale

    model.damping_on = False
    q0, qd0 = torch.tensor([0.0, 2.5]), torch.tensor([0.3, 1.5])
    qs, qds = rollout(model, q0, qd0, torch.zeros(1000, 2), 0.01)
    H = model.energy(qs, qds)
    T0 = (model.energy(q0, qd0) - model.structure(q0[1])["V"]).abs()
    assert (H - H[0]).abs().max() < 1e-6 * T0


def test_3_analytical_structure_reproduces_analytical_model():
    p = load_params()
    q, qd, tau = random_states(5000, seed=2)
    th = q[:, 1]
    a = p.mpl
    ones = torch.ones_like(th)
    qdd, _ = lagrangian_dynamics(p.M_tot * ones, a * torch.cos(th), p.J * ones,
                                 0 * ones, -a * torch.sin(th), 0 * ones, -a * p.g * torch.sin(th),
                                 torch.tensor([p.b_x, p.b_th]), q, qd, tau)
    ref = AnalyticalModel(p).qdd(q, qd, tau)
    assert torch.allclose(qdd, ref, atol=1e-11, rtol=0)


def test_4_inverse_of_forward_is_identity():
    model = random_delan(3)
    q, qd, tau = random_states(3000, seed=3)
    tau_hat, _ = model.inverse(q, qd, model.qdd(q, qd, tau))
    assert torch.allclose(tau_hat, tau, atol=1e-9)


def test_6_forward_mode_derivatives_match_autograd_and_work_without_grad():
    model = random_delan(4)
    th = torch.linspace(-7, 7, 501, requires_grad=True)
    s = model.structure(th)
    for key in ("M11", "M12", "M22", "V"):
        (ref,) = torch.autograd.grad(s[key].sum(), th, retain_graph=True)
        assert torch.allclose(s["d" + key], ref, atol=1e-10), key
    q, qd, tau = random_states(100, seed=4)
    with torch.no_grad():
        a = model.qdd(q, qd, tau)
    assert torch.allclose(a, model.qdd(q, qd, tau))


@pytest.mark.parametrize("kind", ["mlp", "delan"])
def test_5_overfits_one_batch(kind):
    """Plumbing check: each model can fit 256 rows of real data closely."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cartpole.dataset import build
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        d = build("A", 1, seed=0)
        tr, st = d["train"], d["stats"]
        idx = torch.arange(256)
        q, qd, tau, y = (tr[k][idx] for k in ("q", "qd", "tau", "qdd"))
        model = MLPModel(st) if kind == "mlp" else DeLaN()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(3000):
            loss = (((model.qdd(q, qd, tau) - y) / st["qdd_std"]) ** 2).sum(-1).mean()
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Plumbing, not speed: at a constant lr of 1e-3 DeLaN's loss is non-monotone
        # (ISSUES T1), so require a 1000x drop and < 1 % unexplained variance.
        assert loss.item() < 1e-2 and loss.item() < 1e-3 * first
    finally:
        torch.set_default_dtype(torch.float64)
