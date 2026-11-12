"""Phase 2: generate the datasets.

Pools (split by episode, never by row):
  R  random     u ~ U[-1,1] iid per step          train 20 / val 5 / test 10
  E  pumping    energy pumping + OU noise         train 20 / val 5 / test 10
  U  unforced   u = 0, dof_damping = 0, 30 initial states (energy-drift ground truth)

Usage:  python scripts/p2_collect.py
Outputs: data/{R,E}_{train,val,test}.npz, data/U.npz,
         results/p2_dataset.json, figures/p2_coverage.*
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cartpole.data import (EnergyPumpingPolicy, flatten, random_policy, run_episode,  # noqa: E402
                           save_episodes, undamped_hook, unforced_initial_states)
from cartpole.integrators import wrap_angle  # noqa: E402
from cartpole.params import load_params, make_env  # noqa: E402
from cartpole.plotting import DATA_DIR, RES_DIR, savefig, setup_style  # noqa: E402

SPLITS = {
    "R_train": ("random", range(0, 20)),
    "R_val": ("random", range(100, 105)),
    "R_test": ("random", range(200, 210)),
    "E_train": ("pumping", range(300, 320)),
    "E_val": ("pumping", range(400, 405)),
    "E_test": ("pumping", range(500, 510)),
}
U_SEED = 600
COS_HANGING = -0.8   # "hanging" for the OOR-velocity set: within 36.9 deg of straight down


def split_stats(eps, p):
    d = flatten(eps)
    th, thd = d["q"][:, 1], d["qd"][:, 1]
    phi = wrap_angle(th - np.pi)
    rot = [abs(e["qpos"][-1, 1] - e["qpos"][0, 1]) / (2 * np.pi) for e in eps]
    return dict(
        n_episodes=len(eps), n_rows=int(len(th)),
        limit_frac=float(d["limit_active"].mean()),
        max_abs_x=float(np.abs(d["q"][:, 0]).max()),
        max_abs_xd=float(np.abs(d["qd"][:, 0]).max()),
        max_abs_phi_from_hanging=float(np.abs(phi).max()),
        max_abs_thd=float(np.abs(thd).max()),
        max_cos_theta=float(np.cos(th).max()),
        frac_above_horizontal=float((np.cos(th) > 0).mean()),
        qdd_std=d["qdd"].std(0).tolist(),
        qdd_max_abs=np.abs(d["qdd"]).max(0).tolist(),
        u_saturated_frac=float((np.abs(d["u"]) >= 0.999).mean()),
        net_turns_max=float(max(rot)),
    )


def internal_energy(q, qd, p):
    """Routh-reduced energy above hanging: H - p_x^2/(2 M_t) + m_p g l.
    = 1/2 J_eff(th) thd^2 + m_p g l (1 + cos th),  J_eff = J - (m_p l cos th)^2 / M_t.
    Equals 2 m_p g l at upright rest."""
    th, thd = q[..., 1], qd[..., 1]
    J_eff = p.J - (p.mpl * np.cos(th)) ** 2 / p.M_tot
    return 0.5 * J_eff * thd ** 2 + p.mpl * p.g * (1 + np.cos(th))


def main():
    setup_style()
    import matplotlib.pyplot as plt

    p = load_params()
    agl = p.mpl * p.g
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = {"splits": {}}
    all_eps = {}
    t0 = time.time()

    # ---- R and E pools ---------------------------------------------------
    for name, (kind, seeds) in SPLITS.items():
        eps, erefs = [], []
        for s in seeds:
            env = make_env(seed=s)
            if kind == "random":
                pol = random_policy(s)
            else:
                pol = EnergyPumpingPolicy(p, s)
            eps.append(run_episode(env, pol))
            if kind == "pumping":
                erefs.append(pol.E_refs)
        meta = dict(pool=kind, seeds=list(seeds))
        if erefs:
            meta["E_ref_over_mgl"] = np.array(erefs)
        save_episodes(DATA_DIR / f"{name}.npz", eps, meta)
        all_eps[name] = eps
        res["splits"][name] = split_stats(eps, p)
        print(f"{name}: {len(eps)} episodes ({time.time() - t0:.0f}s)")

    # ---- U pool: unforced, undamped ---------------------------------------
    states = unforced_initial_states(p, seed=U_SEED)
    u_eps, groups = [], []
    for i, st in enumerate(states):
        env = make_env(seed=U_SEED + i)
        ep = run_episode(env, lambda ph, k: 0.0, physics_hook=undamped_hook(st["qpos"], st["qvel"]))
        u_eps.append(ep)
        groups.append(st["group"])
    save_episodes(DATA_DIR / "U.npz", u_eps, dict(pool="unforced_undamped", groups=np.array(groups)))
    u_stats = {}
    for g in ("small", "large", "rotating"):
        sel = [e for e, gg in zip(u_eps, groups) if gg == g]
        H = [e["energy"].sum(-1) for e in sel]
        u_stats[g] = dict(
            n=len(sel),
            limit_ever_active=bool(any(e["limit_active"].any() for e in sel)),
            max_abs_x=float(max(np.abs(e["qpos"][:, 0]).max() for e in sel)),
            internal_energy_over_mgl=[float(internal_energy(e["qpos"][0], e["qvel"][0], p) / agl) for e in sel],
            mujoco_max_drift_over_mgl=float(max(np.abs(h - h[0]).max() for h in H) / agl),
        )
    res["U"] = u_stats

    # ---- OOR definitions (from the data) ----------------------------------
    R = flatten(all_eps["R_train"])
    cos_max_R = float(np.cos(R["q"][:, 1]).max())
    thd_max_R = float(np.abs(R["qd"][:, 1]).max())
    Et = flatten(all_eps["E_test"])
    free = ~Et["limit_active"]
    c, w = np.cos(Et["q"][:, 1]), np.abs(Et["qd"][:, 1])
    oor_angle = (c > 0) & free
    oor_vel = (c < COS_HANGING) & (w > thd_max_R) & free
    in_support = (c <= cos_max_R) & (w <= thd_max_R) & free
    res["oor"] = dict(
        R_train_max_cos_theta=cos_max_R, R_train_max_abs_thd=thd_max_R,
        definition_angle="cos(theta) > 0 (pole above horizontal)",
        definition_velocity=f"cos(theta) < {COS_HANGING} and |thd| > R_train max |thd|",
        E_test_rows=int(len(c)), E_test_oor_angle_rows=int(oor_angle.sum()),
        E_test_oor_velocity_rows=int(oor_vel.sum()), E_test_in_R_support_rows=int(in_support.sum()),
    )

    # ---- random-walk energy theory ----------------------------------------
    # Linearised about hanging: phi'' + w^2 phi = beta F, beta = m_p l / (M_t J_eff0).
    # iid force with variance s^2 held for h: E[internal energy](t) = 1/2 J_eff0 beta^2 s^2 h t.
    J_eff0 = p.J - p.mpl ** 2 / p.M_tot
    beta = p.mpl / (p.M_tot * J_eff0)
    s2 = p.gear ** 2 / 3.0
    rate = 0.5 * J_eff0 * beta ** 2 * s2 * p.dt              # J/s
    R_all = all_eps["R_train"] + all_eps["R_val"] + all_eps["R_test"]
    Ein = np.stack([internal_energy(e["qpos"], e["qvel"], p) for e in R_all])
    tt = np.arange(Ein.shape[1]) * p.dt
    pinned = np.stack([e["limit_active"] for e in R_all])
    slope = float((tt @ Ein.mean(0)) / (tt @ tt))
    # Same average, but each episode only until its first rail contact (the
    # theory has no constraint impulses). Keep times with >= 5 episodes left.
    first = np.array([np.argmax(r) if r.any() else len(r) for r in pinned])
    alive = tt[None, :] < first[:, None] * p.dt
    n_alive = alive.sum(0)
    ok = n_alive >= 5
    m_pre = np.where(ok, (Ein * alive).sum(0) / np.maximum(n_alive, 1), np.nan)
    slope_pre = float((tt[ok] @ m_pre[ok]) / (tt[ok] @ tt[ok]))
    res["random_walk"] = dict(
        J_eff0=J_eff0, beta=beta, force_variance=s2, n_episodes=len(R_all),
        predicted_rate_J_per_s=rate, measured_rate_J_per_s=slope,
        measured_rate_before_first_rail_contact_J_per_s=slope_pre,
        rail_contact_time_s_median=float(np.median(first) * p.dt),
        predicted_time_to_upright_energy_s=float(2 * agl / rate),
        predicted_energy_at_10s_over_upright=float(rate * 10 / (2 * agl)),
        measured_mean_energy_at_10s_over_upright=float(Ein[:, -1].mean() / (2 * agl)),
        R_train_frac_time_pinned_at_rail=float(pinned.mean()),
    )

    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "p2_dataset.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax = axes[0]
    E = flatten(all_eps["E_train"])
    phiE, phiR = wrap_angle(E["q"][:, 1] - np.pi), wrap_angle(R["q"][:, 1] - np.pi)
    ymax = 1.1 * max(np.abs(E["qd"][:, 1]).max(), thd_max_R)
    ax.axvspan(-np.pi, -np.pi / 2, color="#e7298a", alpha=0.10, lw=0)
    ax.axvspan(np.pi / 2, np.pi, color="#e7298a", alpha=0.10, lw=0, label="OOR-angle: pole above horizontal")
    a_h = np.arccos(-COS_HANGING)
    for sgn in (1, -1):
        ax.fill_between([-a_h, a_h], sgn * thd_max_R, sgn * ymax, color="#1f78b4", alpha=0.12, lw=0,
                        label="OOR-velocity: hanging, $|\\dot\\theta|$ > random-data max" if sgn == 1 else None)
    ax.scatter(phiE[::3], E["qd"][::3, 1], s=1, color="#d95f02", alpha=0.25, rasterized=True)
    ax.scatter(phiR[::3], R["qd"][::3, 1], s=1, color="#1b9e77", alpha=0.4, rasterized=True)
    ax.plot([], [], "o", ms=5, color="#d95f02", label="energy-pumping (E, train)")
    ax.plot([], [], "o", ms=5, color="#1b9e77", label="random actions (R, train)")
    ph = np.linspace(-np.pi, np.pi, 400)
    th = ph + np.pi
    sep = np.sqrt(2 * agl * (1 - np.cos(th)) / (p.J - (p.mpl * np.cos(th)) ** 2 / p.M_tot))
    ax.plot(ph, sep, "k--", lw=1.2, label="separatrix (reduced energy = upright)")
    ax.plot(ph, -sep, "k--", lw=1.2)
    ax.set(xlim=(-np.pi, np.pi), ylim=(-ymax, ymax),
           xlabel=r"$\varphi = \theta - \pi$ [rad]  (0 = hanging, $\pm\pi$ = upright)",
           ylabel=r"$\dot\theta$ [rad/s]", title="(a) State coverage of the training pools")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    ax = axes[1]
    m, sd = Ein.mean(0) / (2 * agl), Ein.std(0) / (2 * agl)
    ax.fill_between(tt, np.maximum(m - sd, 0), m + sd, color="#1b9e77", alpha=0.2, lw=0, label="±1 std")
    ax.plot(tt, m, color="#1b9e77", lw=2, label=f"random actions: mean of {len(R_all)} episodes")
    ax.plot(tt, m_pre / (2 * agl), color="#7570b3", lw=2,
            label="same, each episode only until first rail contact")
    ax.plot(tt, rate * tt / (2 * agl), "k--", lw=1.5,
            label=f"linear theory: $E = {rate:.4f}\\,t$ J")
    t_up = 2 * agl / rate
    ax.text(0.97, 0.04, f"upright needs 1.0\n→ ≈ {t_up:.0f} s at the theoretical rate\n(episode = 10 s)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set(xlabel="time [s]", ylabel=r"internal energy / $2m_pgl$ (1 = upright)",
           title="(b) Random forcing: energy random walk", ylim=(0, 0.4))
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    bins = np.linspace(-1, 1, 41)
    ax.hist(np.cos(R["q"][:, 1]), bins=bins, density=True, alpha=0.6, color="#1b9e77", label="random (R)")
    ax.hist(np.cos(E["q"][:, 1]), bins=bins, density=True, alpha=0.6, color="#d95f02", label="energy-pumping (E)")
    ax.axvline(0, color="#e7298a", lw=1.2, ls=":")
    ax.set_yscale("log")
    ax.set(xlabel=r"$\cos\theta$  (−1 hanging, +1 upright)", ylabel="density (log)",
           title=r"(c) Distribution of $\cos\theta$")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    savefig(fig, "p2_coverage")


if __name__ == "__main__":
    main()
