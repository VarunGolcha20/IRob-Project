"""Phase 3 diagnostics: learned structure, parameter identification, training curves.

Parameter identification (PLAN 3.6). Each learned quantity is fitted by least
squares on its *physical* basis, evaluated at the training angles of that run
(so the fit uses only states the model was trained on):
    M11(th)      ~ c                 -> m_c + m_p
    M12(th)      ~ alpha cos th      -> m_p l   ("inertial" estimate)
    M22(th)      ~ c                 -> J
    dV/dth (th)  ~ beta (-g sin th)  -> m_p l   ("gravitational" estimate)
The gravity torque dV/dth is fitted rather than V, because V is defined only
up to a constant. A free constant in the M12 fit is not used: near hanging
(regime A) cos th ~ -1, so cos th and 1 are almost collinear and the split
between them would be arbitrary.

Usage:  python scripts/p3_structure.py
Outputs: results/p3_identification.json, figures/p3_learned_structure.*, figures/p3_training_curves.*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole.data import SUPPORT_MIN_COUNT  # noqa: E402
from cartpole.dataset import build  # noqa: E402
from cartpole.params import load_params  # noqa: E402
from cartpole.plotting import MODEL_COLORS, RES_DIR, WEIGHT_DIR, savefig, setup_style  # noqa: E402
from cartpole.train import load_model  # noqa: E402

torch.set_default_dtype(torch.float64)


def lstsq1(basis, y):
    """Least-squares coefficient of y on a single basis function, and the relative residual."""
    c = float(basis @ y / (basis @ basis))
    res = np.sqrt(np.mean((y - c * basis) ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-30)
    return c, float(res)


def identify(model, th: np.ndarray, p) -> dict:
    with torch.no_grad():
        s = {k: v.numpy() for k, v in model.structure(torch.as_tensor(th)).items()}
    a_in, r12 = lstsq1(np.cos(th), s["M12"])
    a_gr, rV = lstsq1(-p.g * np.sin(th), s["dV"])
    true = dict(M_tot=p.M_tot, mpl=p.mpl, J=p.J, b_x=p.b_x, b_th=p.b_th)
    est = dict(M_tot=float(s["M11"].mean()), mpl_inertial=a_in, J=float(s["M22"].mean()),
               mpl_gravitational=a_gr, b_x=float(model.damping[0]), b_th=float(model.damping[1]))
    rel = dict(M_tot=est["M_tot"] / true["M_tot"] - 1, mpl_inertial=a_in / p.mpl - 1,
               J=est["J"] / true["J"] - 1, mpl_gravitational=a_gr / p.mpl - 1)
    return dict(estimate=est, rel_error=rel, fit_rel_residual=dict(M12=r12, dV=rV),
                inertial_vs_gravitational=a_in / a_gr - 1 if a_gr != 0 else None,
                M11_rel_spread=float(s["M11"].std() / s["M11"].mean()),
                M22_rel_spread=float(s["M22"].std() / s["M22"].mean()))


def theta_support(th_train: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Bins of wrapped theta that hold >= SUPPORT_MIN_COUNT training rows."""
    w = np.arctan2(np.sin(th_train), np.cos(th_train))
    h, _ = np.histogram(w, bins=bins)
    return h >= SUPPORT_MIN_COUNT


def main():
    setup_style()
    import matplotlib.pyplot as plt

    p = load_params()
    info = json.loads((RES_DIR / "p3_training.json").read_text())
    lam = info["lam_inv"]
    suffix = f"_inv{lam:g}" if lam > 0 else ""

    # ---- identification for every DeLaN run ---------------------------------
    ident = {}
    for r in info["runs"]:
        if r["kind"] != "delan":
            continue
        model, ck = load_model(WEIGHT_DIR / f"{r['name']}.pt")
        d = build(r["regime"], r["N"], r["seed"], dtype=torch.float64)
        th = d["train"]["q"][:, 1].numpy()
        ident[r["name"]] = dict(regime=r["regime"], N=r["N"], seed=r["seed"], **identify(model, th, p))
        # full circle (meaningful only where trained)
        ident[r["name"]]["full_circle"] = identify(model, np.linspace(-np.pi, np.pi, 721), p)["rel_error"]

    summary = {}
    for reg, N in sorted({(v["regime"], v["N"]) for v in ident.values()}):
        rows = [v for v in ident.values() if v["regime"] == reg and v["N"] == N]
        summary[f"{reg}{N}"] = {
            k: dict(mean=float(np.mean([r["rel_error"][k] for r in rows])),
                    std=float(np.std([r["rel_error"][k] for r in rows])))
            for k in rows[0]["rel_error"]}
        summary[f"{reg}{N}"]["inertial_vs_gravitational_mean"] = float(np.mean([r["inertial_vs_gravitational"] for r in rows]))
        summary[f"{reg}{N}"]["damping_mean"] = [float(np.mean([r["estimate"]["b_x"] for r in rows])),
                                                float(np.mean([r["estimate"]["b_th"] for r in rows]))]
    out = dict(true=dict(M_tot=p.M_tot, mpl=p.mpl, J=p.J, b_x=p.b_x, b_th=p.b_th),
               lam_inv=lam, summary_rel_error_on_training_angles=summary, runs=ident)
    (RES_DIR / "p3_identification.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- learned structure figure ------------------------------------------
    def best_seed(kind, reg, N):
        rows = [r for r in info["runs"] if r["kind"] == kind and r["regime"] == reg and r["N"] == N]
        return min(rows, key=lambda r: r["best_val"])

    grid = np.linspace(-np.pi, np.pi, 721)
    bins = np.linspace(-np.pi, np.pi, 64)
    truth = dict(M11=np.full_like(grid, p.M_tot), M12=p.mpl * np.cos(grid), M22=np.full_like(grid, p.J),
                 dV=-p.mpl * p.g * np.sin(grid))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    titles = dict(M11=r"$M_{11}(\theta)$ [kg]", M12=r"$M_{12}(\theta)$ [kg m]",
                  M22=r"$M_{22}(\theta)$ [kg m$^2$]", dV=r"gravity torque $\partial V/\partial\theta$ [N m]")
    styles = {("A", 20): dict(color="#7570b3", ls="-"), ("B", 20): dict(color="#d95f02", ls="-"),
              ("A", 1): dict(color="#7570b3", ls=":")}
    supA = None
    for (reg, N), st in styles.items():
        r = best_seed("delan", reg, N)
        model, _ = load_model(WEIGHT_DIR / f"{r['name']}.pt")
        with torch.no_grad():
            s = {k: v.numpy() for k, v in model.structure(torch.as_tensor(grid)).items()}
        if (reg, N) == ("A", 20):
            d = build("A", 20, r["seed"])
            supA = theta_support(d["train"]["q"][:, 1].numpy(), bins)
        for key, ax in zip(titles, axes.flat):
            ax.plot(grid, s[key], lw=2, label=f"DeLaN, regime {reg}, N={N} (seed {r['seed']})", **st)
    for key, ax in zip(titles, axes.flat):
        ax.plot(grid, truth[key], color="k", lw=1.5, ls="--", label="true (MuJoCo)")
        for i, ok in enumerate(supA):
            if ok:
                ax.axvspan(bins[i], bins[i + 1], color="#7570b3", alpha=0.08, lw=0)
        ax.set_title(titles[key])
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$" + "\n(hanging)", r"$-\pi/2$", "0\n(upright)", r"$\pi/2$", r"$\pi$" + "\n(hanging)"])
    axes[0, 0].legend(loc="best", fontsize=9)
    axes[1, 0].set_xlabel(r"$\theta$ [rad]")
    axes[1, 1].set_xlabel(r"$\theta$ [rad]")
    fig.suptitle("Learned Lagrangian structure vs truth (shaded: angles covered by regime-A training data)", y=1.0)
    fig.tight_layout()
    savefig(fig, "p3_learned_structure")

    # ---- training curves ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, reg in zip(axes, ("A", "B")):
        for kind in ("mlp", "delan"):
            for N, ls in ((1, ":"), (5, "--"), (20, "-")):
                rows = [r for r in info["runs"] if r["kind"] == kind and r["regime"] == reg and r["N"] == N]
                if not rows:
                    continue
                ck = torch.load(WEIGHT_DIR / f"{rows[0]['name']}.pt", weights_only=False)
                h = ck["history"]
                ax.semilogy(h["step"], h["val"], ls=ls, color=MODEL_COLORS[kind], lw=1.8,
                            label=f"{kind.upper() if kind == 'mlp' else 'DeLaN'}, N={N} (seed 0)")
        ax.set(xlabel="gradient step", title=f"Regime {reg}: validation loss")
        ax.legend(fontsize=9, ncol=2)
    axes[0].set_ylabel("normalised forward loss (validation)")
    fig.tight_layout()
    savefig(fig, "p3_training_curves")


if __name__ == "__main__":
    main()
