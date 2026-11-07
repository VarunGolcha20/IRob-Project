"""Phase 1: validate the analytical Euler-Lagrange model against MuJoCo.

Checks (along recorded trajectories):
  1. M(q)               vs mujoco.mj_fullM
  2. C qd + g           vs qfrc_bias
  3. -D qd              vs qfrc_passive
  4. forward dynamics   vs qacc            (tau = qfrc_actuator + qfrc_constraint)
  5. inverse dynamics   vs mujoco.mj_inverse (qfrc_inverse)
  6. our RK4 rollout    vs MuJoCo's RK4 rollout
  7. hand-derived capsule inertia vs body_inertia
plus: energy formula vs data.energy, linearisations, M conditioning.

Usage:  python scripts/p1_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel, thin_rod_model  # noqa: E402
from cartpole.data import random_policy, run_episode  # noqa: E402
from cartpole.integrators import rollout, wrap_angle  # noqa: E402
from cartpole.params import capsule_inertia_perp, load_params, make_env  # noqa: E402
from cartpole.plotting import RES_DIR, savefig, setup_style  # noqa: E402

torch.set_default_dtype(torch.float64)
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731


def collect():
    """5 random episodes + 1 resonant-sinusoid episode (large nonlinear swings)."""
    eps, names = [], []
    for s in range(5):
        env = make_env(seed=1000 + s)
        eps.append(run_episode(env, random_policy(1000 + s)))
        names.append(f"random_{s}")
    env = make_env(seed=1100)
    p = load_params(env.physics)
    omega = np.sqrt(p.mpl * p.g * p.M_tot / (p.J * p.M_tot - p.mpl ** 2))
    eps.append(run_episode(env, lambda ph, k: 0.3 * np.sin(omega * k * p.dt)))
    names.append("resonant_sine")
    return eps, names


def inverse_mujoco(physics, qpos, qvel, qacc):
    """qfrc_inverse from mj_inverse on a scratch MjData (does not touch the env)."""
    m = physics.model.ptr
    d = mujoco.MjData(m)
    out = np.zeros_like(qpos)
    for i in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[i], qvel[i], qacc[i]
        mujoco.mj_inverse(m, d)
        out[i] = d.qfrc_inverse
    return out


def main():
    setup_style()
    import matplotlib.pyplot as plt

    env = make_env()
    p = load_params(env.physics)
    model = AnalyticalModel(p)
    res = {"params": p.as_dict()}

    # ---- 7. capsule inertia --------------------------------------------
    m = env.physics.model
    gid = m.name2id("pole_1", "geom")
    r, half = m.geom_size[gid][:2]
    I_hand = capsule_inertia_perp(p.m_p, r, 2 * half)
    rod = thin_rod_model(p)
    res["inertia"] = dict(I_c_mujoco=p.I_c, I_c_capsule_formula=I_hand,
                          rel_err=abs(I_hand - p.I_c) / p.I_c,
                          J=p.J, J_thin_rod=rod.p.J,
                          thin_rod_J_rel_err=(rod.p.J - p.J) / p.J)

    # ---- data -------------------------------------------------------------
    eps, names = collect()
    allr = {k: np.concatenate([e[k] for e in eps]) for k in eps[0] if k not in ("qpos_final", "qvel_final")}
    q, qd = T(allr["qpos"]), T(allr["qvel"])
    tau_act, tau_con = T(allr["qfrc_actuator"]), T(allr["qfrc_constraint"])
    qacc = T(allr["qacc"])
    lim = allr["limit_active"]

    # ---- 1-5 residuals ----------------------------------------------------
    r1 = (model.mass_matrix(q) - T(allr["fullM"]).reshape(-1, 2, 2)).reshape(-1, 4)
    r2 = model.bias(q, qd) - T(allr["qfrc_bias"])
    r3 = -model.damping_force(qd) - T(allr["qfrc_passive"])
    r4 = model.qdd(q, qd, tau_act + tau_con) - qacc
    qfrc_inv = T(inverse_mujoco(env.physics, allr["qpos"], allr["qvel"], allr["qacc"]))
    tau_hat = model.inverse(q, qd, qacc)
    r5a = qfrc_inv - tau_act                       # MuJoCo's own consistency
    r5b = (tau_hat - tau_con) - qfrc_inv           # ours vs mj_inverse
    r_e = model.energy(q, qd, mujoco_offset=True) - T(allr["energy"]).sum(-1)

    checks = {
        "1_mass_matrix": r1, "2_bias": r2, "3_passive": r3,
        "4_forward_qacc": r4, "5_inverse_vs_mj_inverse": r5b,
        "5a_mj_inverse_vs_actuator": r5a, "energy": r_e,
    }
    res["residual_max_abs"] = {k: float(v.abs().max()) for k, v in checks.items()}
    res["n_rows"] = int(len(lim))
    res["n_limit_rows"] = int(lim.sum())
    res["limit_rows_residual_max_abs"] = {
        k: float(v[torch.as_tensor(lim)].abs().max()) if lim.any() else None
        for k, v in checks.items()}
    res["free_rows_residual_max_abs"] = {
        k: float(v[torch.as_tensor(~lim)].abs().max()) for k, v in checks.items()}

    # 4b. Limit-row residual comes from the constraint solver's tolerance
    # (default 1e-8): rerun one episode with tolerance 1e-15.
    tol_test = {}
    for tol in (None, 1e-15):
        e = make_env(seed=1001)
        if tol is not None:
            e.physics.model.opt.tolerance = tol
            e.physics.model.opt.iterations = 1000
        ep = run_episode(e, random_policy(1001))
        rr = (model.qdd(T(ep["qpos"]), T(ep["qvel"]),
                        T(ep["qfrc_actuator"] + ep["qfrc_constraint"])) - T(ep["qacc"])).abs().max(-1).values.numpy()
        tol_test[str(tol or e.physics.model.opt.tolerance)] = dict(
            limit_rows=float(rr[ep["limit_active"]].max()), free_rows=float(rr[~ep["limit_active"]].max()))
    res["4b_qacc_residual_vs_solver_tolerance"] = tol_test
    # Scale references, so residuals can be read relative to signal size.
    res["signal_scale"] = {
        "qacc_abs_max": allr["qacc"].__abs__().max(0).tolist(),
        "qfrc_actuator_abs_max": float(np.abs(allr["qfrc_actuator"][:, 0]).max()),
        "qfrc_constraint_abs_max": float(np.abs(allr["qfrc_constraint"]).max()),
    }

    # ---- 6. RK4 rollout vs MuJoCo -----------------------------------------
    roll = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, ep in zip(names, eps):
        n = len(ep["u"])
        first_lim = np.argmax(ep["limit_active"]) if ep["limit_active"].any() else n
        if first_lim < 50:
            continue
        tau = torch.zeros(first_lim, 2)
        tau[:, 0] = p.gear * T(ep["u"][:first_lim])
        qs, qds = rollout(model, T(ep["qpos"][0]), T(ep["qvel"][0]), tau, p.dt)
        # recorded states s_0..s_{first_lim-1}; rollout states 0..first_lim
        ref_q = np.vstack([ep["qpos"][:first_lim],
                           ep["qpos_final"][None] if first_lim == n else np.empty((0, 2))])
        ref_qd = np.vstack([ep["qvel"][:first_lim],
                            ep["qvel_final"][None] if first_lim == n else np.empty((0, 2))])
        k = len(ref_q)
        eq = (qs[:k].numpy() - ref_q)
        eq[:, 1] = wrap_angle(eq[:, 1])
        eqd = qds[:k].numpy() - ref_qd
        roll[name] = dict(steps=int(k - 1), seconds=float((k - 1) * p.dt),
                          max_abs_q=np.abs(eq).max(0).tolist(),
                          max_abs_qd=np.abs(eqd).max(0).tolist(),
                          theta_range=[float(ep["qpos"][:k, 1].min()), float(ep["qpos"][:k, 1].max())])
        tt = np.arange(k) * p.dt
        axes[0].semilogy(tt, np.abs(eq[:, 0]) + 1e-18, label=name)
        axes[1].semilogy(tt, np.abs(eq[:, 1]) + 1e-18, label=name)
    res["rk4_rollout_vs_mujoco"] = roll
    axes[0].set(xlabel="time [s]", ylabel=r"$|x_{RK4} - x_{MuJoCo}|$ [m]", title="Cart position")
    axes[1].set(xlabel="time [s]", ylabel=r"$|\theta_{RK4} - \theta_{MuJoCo}|$ [rad]", title="Pole angle")
    axes[1].legend(loc="lower right", ncol=2)
    for ax in axes:
        ax.set_ylim(5e-19, 1e-13)
    fig.suptitle("Our RK4 + analytical model vs MuJoCo, until first rail-limit contact "
                 "(values at $10^{-18}$ are exact zeros)", y=1.03)
    savefig(fig, "p1_rk4_vs_mujoco")

    # ---- residual histogram figure ----------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    titles = ["1: $M(q)$ vs mj_fullM", "2: $C\\dot{q}+g$ vs qfrc_bias",
              "3: $-D\\dot{q}$ vs qfrc_passive", "4: $\\ddot{q}$ vs qacc",
              "5: $\\hat{\\tau}$ vs mj_inverse"]
    free = torch.as_tensor(~lim)
    for ax, key, title in zip(axes, list(checks)[:5], titles):
        v = checks[key][free].abs().flatten().numpy()
        v = np.log10(np.maximum(v, 1e-19))      # exact zeros land in the -19 bin
        ax.hist(v, bins=48, range=(-19.25, -7.25), color="#1b9e77")
        ax.set(title=title, xlabel=r"$\log_{10}|\mathrm{residual}|$", xlim=(-19.5, -7))
        ax.set_xticks([-19, -16, -13, -10])
        ax.set_xticklabels(["exact\n0", "-16", "-13", "-10"])
    axes[0].set_ylabel("count (entries)")
    fig.suptitle("Analytical Euler–Lagrange model vs MuJoCo: residuals over "
                 f"{int((~lim).sum())} recorded states with the rail limit inactive", y=1.04)
    savefig(fig, "p1_residuals")

    # ---- linearisations, conditioning, time scales -------------------------
    def f(s):
        qq, vv = s[:2], s[2:]
        return torch.cat([vv, model.qdd(qq, vv, torch.zeros(2))])

    lin = {}
    for name, th in (("hanging", np.pi), ("upright", 0.0)):
        A = torch.autograd.functional.jacobian(f, T([0.0, th, 0.0, 0.0]))
        ev = np.linalg.eigvals(A.numpy())
        lin[name] = sorted([[float(e.real), float(e.imag)] for e in ev])
    omega = np.sqrt(p.mpl * p.g * p.M_tot / (p.J * p.M_tot - p.mpl ** 2))
    eig = np.linalg.eigvalsh(model.mass_matrix(T([[0.0, 0.0]])).numpy()[0])
    res["linearisation_eigs"] = lin
    res["omega_formula"] = float(omega)
    res["period_s"] = float(2 * np.pi / omega)
    res["error_doubling_time_upright_s"] = float(np.log(2) / omega)
    res["M_eig_at_cos1"] = eig.tolist()
    res["M_condition"] = float(eig[-1] / eig[0])
    res["damping_time_constants_s"] = {"cart": p.M_tot / p.b_x, "pole": p.J / p.b_th}

    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "p1_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
