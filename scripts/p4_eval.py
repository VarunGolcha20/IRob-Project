"""Phase 4: open-loop evaluation of every trained model (PLAN Phase 4).

Torch-free and MuJoCo-free: models run through cartpole.npdyn in float64, and
the ground truth is the recorded MuJoCo data (test pools R/E/S, unforced pool U).

Metrics (definitions in PLAN Phase 4 and docs/PART4_EVAL.md):
  one_step   nRMSE of qdd per axis and region; normaliser = per-axis std of qdd
             over all test rows (R_test + E_test + S_test, limit rows dropped).
             Also the state error after one RK4 step against the next recorded state.
  rollout    open-loop replay of recorded tau from every 25th step; RMSE of wrapped
             theta and of x over [0, T] for T = 0.5/1/2 s; median + IQR over windows.
             Windows that touch the rail limit are excluded (ISSUES A2).
  vpt        valid-prediction time: first t with |wrap(dtheta)| > 0.1 rad (capped at 2 s).
  drift      U pool, tau = 0, 10 s: max_t |H_true(s_hat) - H_true(s0)| / (m_p g l) per
             group; DeLaN with D = 0 and with learned D; DeLaN's own H drift; p_x drift.
  oor        one-step error on OOR-angle / OOR-velocity; error vs Chebyshev distance
             (in bins) to the model's density support.
  inverse    DeLaN tau_hat vs recorded tau; theta-row residual (should be 0).
  compute    microseconds per batched qdd call (batch 512, numpy float64).

Usage:  python scripts/p4_eval.py              (evaluate + plot, ~5 min)
        python scripts/p4_eval.py --plots-only (re-plot from results/p4_eval.json)
Outputs: results/p4_eval.json, figures/p4_*.{png,pdf}
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cartpole import npdyn as nd  # noqa: E402

H = 0.01
HORIZONS = {"0.5s": 50, "1s": 100, "2s": 200}
STRIDE = 25
VPT_TOL = 0.1
SUP_BIN_TH, SUP_BIN_THD, SUP_THD_MAX, SUP_MIN = 0.1, 0.25, 12.0, 5   # = cartpole.data
TEST_POOLS = ("R_test", "E_test", "S_test")
OOR_THD = 5.291712923553994      # R_train max |thd| (results/p2_dataset.json)
FIG_DIR = nd.ROOT / "figures"


# ---------------------------------------------------------------------------
# support
# ---------------------------------------------------------------------------
_ETH = np.arange(-np.pi, np.pi + 1e-9, SUP_BIN_TH)
_ETHD = np.arange(-SUP_THD_MAX, SUP_THD_MAX + 1e-9, SUP_BIN_THD)


def support_hist(q, qd):
    Hh, _, _ = np.histogram2d(nd.wrap(q[:, 1]), qd[:, 1], bins=[_ETH, _ETHD])
    return Hh >= SUP_MIN


def bin_index(q, qd):
    i = np.clip(np.digitize(nd.wrap(q[:, 1]), _ETH) - 1, 0, len(_ETH) - 2)
    j = np.digitize(qd[:, 1], _ETHD) - 1
    return i, j


def support_distance(mask, q, qd):
    """Chebyshev distance (in bins, theta periodic) from each row's bin to the nearest supported bin."""
    ni, nj = mask.shape
    si, sj = np.nonzero(mask)
    i, j = bin_index(q, qd)
    di = np.abs(i[:, None] - si[None, :])
    di = np.minimum(di, ni - di)
    dj = np.abs(j[:, None] - sj[None, :])
    return np.max(np.stack([di, dj]), 0).min(1)


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------
def load_tests():
    pools = {s: nd.load_episodes(s) for s in TEST_POOLS}
    rows = {s: nd.flatten(pools[s], drop_limit=True) for s in TEST_POOLS}
    allq = np.concatenate([rows[s]["qdd"] for s in TEST_POOLS])
    norm = allq.std(0)
    # one-step state targets: next row in the same episode, both limit-free
    for s in TEST_POOLS:
        q, qd, tau, nq, nqd = [], [], [], [], []
        for e in pools[s]:
            ok = ~(e["limit_active"][:-1] | e["limit_active"][1:])
            q.append(e["qpos"][:-1][ok]); qd.append(e["qvel"][:-1][ok])
            tau.append(e["qfrc_actuator"][:-1][ok])
            nq.append(e["qpos"][1:][ok]); nqd.append(e["qvel"][1:][ok])
        rows[s]["step"] = {k: np.concatenate(v) for k, v in
                           dict(q=q, qd=qd, tau=tau, nq=nq, nqd=nqd).items()}
    return pools, rows, norm


def regions(rows):
    """Boolean masks over the concatenated test rows (R, E, S order)."""
    q = np.concatenate([rows[s]["q"] for s in TEST_POOLS])
    qd = np.concatenate([rows[s]["qd"] for s in TEST_POOLS])
    pool = np.concatenate([np.full(len(rows[s]["q"]), k) for k, s in enumerate(TEST_POOLS)])
    c = np.cos(q[:, 1])
    return q, qd, pool, {
        "all": np.ones(len(q), bool),
        "hanging": c < -0.8,
        "mid": (c >= -0.8) & (c <= 0.8),
        "upright": c > 0.8,
        "R_test": pool == 0,
        "balance (S_test)": pool == 2,
        "OOR-angle (E_test)": (pool == 1) & (c > 0),
        "OOR-velocity (E_test)": (pool == 1) & (c < -0.8) & (np.abs(qd[:, 1]) > OOR_THD),
    }


def windows(pools, pool):
    """All replay windows of length 200 starting every STRIDE steps, limit-free."""
    T = max(HORIZONS.values())
    q0, qd0, taus, qs, qds = [], [], [], [], []
    for e in pools[pool]:
        n = len(e["u"])
        qpos = np.vstack([e["qpos"], e["qpos_final"]]); qvel = np.vstack([e["qvel"], e["qvel_final"]])
        lim = np.append(e["limit_active"], False)
        for t0 in range(0, n - T + 1, STRIDE):
            if lim[t0:t0 + T + 1].any():
                continue
            q0.append(qpos[t0]); qd0.append(qvel[t0])
            taus.append(e["qfrc_actuator"][t0:t0 + T])
            qs.append(qpos[t0:t0 + T + 1]); qds.append(qvel[t0:t0 + T + 1])
    return dict(q0=np.array(q0), qd0=np.array(qd0), tau=np.array(taus).transpose(1, 0, 2),
                q=np.array(qs).transpose(1, 0, 2), qd=np.array(qds).transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def one_step(model, rows, masks, norm, sup_mask=None, q=None, qd=None):
    tau = np.concatenate([rows[s]["tau"] for s in TEST_POOLS])
    y = np.concatenate([rows[s]["qdd"] for s in TEST_POOLS])
    with np.errstate(all="ignore"):
        e = (model.qdd(q, qd, tau) - y) / norm
    out = {}
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        out[name] = dict(n=int(m.sum()), nrmse_xdd=float(np.sqrt(np.mean(e[m, 0] ** 2))),
                         nrmse_thdd=float(np.sqrt(np.mean(e[m, 1] ** 2))))
    if sup_mask is not None:
        i, j = bin_index(q, qd)
        ok = (j >= 0) & (j < sup_mask.shape[1])
        ins = np.zeros(len(q), bool)
        ins[ok] = sup_mask[i[ok], j[ok]]
        for name, m in (("in support", ins), ("out of support", ~ins)):
            if m.sum():
                out[name] = dict(n=int(m.sum()), nrmse_xdd=float(np.sqrt(np.mean(e[m, 0] ** 2))),
                                 nrmse_thdd=float(np.sqrt(np.mean(e[m, 1] ** 2))))
    # state error after one RK4 step
    st = {k: np.concatenate([rows[s]["step"][k] for s in TEST_POOLS]) for k in ("q", "qd", "tau", "nq", "nqd")}
    with np.errstate(all="ignore"):
        nq, nqd = nd.rk4_step(model, st["q"], st["qd"], st["tau"], H)
    out["state_after_1_step_rms"] = dict(
        x=float(np.sqrt(np.mean((nq[:, 0] - st["nq"][:, 0]) ** 2))),
        theta=float(np.sqrt(np.mean(nd.wrap(nq[:, 1] - st["nq"][:, 1]) ** 2))),
        xd=float(np.sqrt(np.mean((nqd[:, 0] - st["nqd"][:, 0]) ** 2))),
        thd=float(np.sqrt(np.mean((nqd[:, 1] - st["nqd"][:, 1]) ** 2))))
    return out, e


def rollout_metrics(model, W):
    with np.errstate(all="ignore"):
        q, _ = nd.rollout(model, W["q0"], W["qd0"], W["tau"], H)
    dth = np.abs(nd.wrap(q[..., 1] - W["q"][..., 1]))
    dth = np.where(np.isfinite(dth), dth, np.pi)
    dx = np.abs(q[..., 0] - W["q"][..., 0])
    dx = np.where(np.isfinite(dx), dx, 10.0)
    out = {}
    for name, T in HORIZONS.items():
        rth = np.sqrt(np.mean(dth[1:T + 1] ** 2, 0))
        rx = np.sqrt(np.mean(dx[1:T + 1] ** 2, 0))
        out[name] = dict(theta_median=float(np.median(rth)), theta_iqr=[float(v) for v in np.percentile(rth, [25, 75])],
                         x_median=float(np.median(rx)), x_iqr=[float(v) for v in np.percentile(rx, [25, 75])])
    bad = dth > VPT_TOL
    first = np.where(bad.any(0), bad.argmax(0), dth.shape[0] - 1) * H
    out["vpt_mean_s"] = float(first.mean())
    out["vpt_median_s"] = float(np.median(first))
    out["n_windows"] = int(dth.shape[1])
    return out


def load_U():
    z = np.load(nd.DATA_DIR / "U.npz")
    groups = [str(g) for g in z["meta/groups"]]
    eps = nd.load_episodes("U")
    return eps, groups


def drift(model, U, p, label_damped=True):
    eps, groups = U
    q0 = np.array([e["qpos"][0] for e in eps]); qd0 = np.array([e["qvel"][0] for e in eps])
    T = len(eps[0]["u"])
    tau = np.zeros((T,) + q0.shape)
    with np.errstate(all="ignore"):
        q, qd = nd.rollout(model, q0, qd0, tau, H)
        Ht = nd.true_energy(p, q, qd)
        dH = np.abs(Ht - Ht[0]) / p.mgl
        dH = np.where(np.isfinite(dH), dH, np.inf).max(0)
        own = None
        if hasattr(model, "energy") and not isinstance(model, nd.Analytical):
            Hm = model.energy(q, qd)
            own = np.abs(Hm - Hm[0]) / p.mgl
            own = np.where(np.isfinite(own), own, np.inf).max(0)
        px = p.M_tot * qd[..., 0] + p.mpl * np.cos(q[..., 1]) * qd[..., 1]
        dpx = np.abs(px - px[0]).max(0)
        # state error vs MuJoCo (undamped, unforced) at 10 s
        qT = np.array([e["qpos_final"] for e in eps])
        th_err = np.abs(nd.wrap(q[-1, :, 1] - qT[:, 1]))
    out = {}
    for g in dict.fromkeys(groups):
        m = np.array([gg == g for gg in groups])
        out[g] = dict(dH_true_median=float(np.median(dH[m])), dH_true_max=float(np.max(dH[m])),
                      px_drift_max=float(np.nanmax(dpx[m])) if np.isfinite(dpx[m]).any() else None,
                      theta_err_10s_median=float(np.nanmedian(th_err[m])) if np.isfinite(th_err[m]).any() else None)
        if own is not None:
            out[g]["dH_own_max"] = float(np.max(own[m]))
    return out


def inverse_metrics(model, rows):
    q = np.concatenate([rows[s]["q"] for s in TEST_POOLS]); qd = np.concatenate([rows[s]["qd"] for s in TEST_POOLS])
    tau = np.concatenate([rows[s]["tau"] for s in TEST_POOLS]); qdd = np.concatenate([rows[s]["qdd"] for s in TEST_POOLS])
    th, _ = model.inverse(q, qd, qdd)
    return dict(tau_x_rmse_N=float(np.sqrt(np.mean((th[:, 0] - tau[:, 0]) ** 2))),
                tau_x_nrmse=float(np.sqrt(np.mean((th[:, 0] - tau[:, 0]) ** 2)) / tau[:, 0].std()),
                tau_theta_rms_Nm=float(np.sqrt(np.mean(th[:, 1] ** 2))))


def timing(model, n=512, reps=200):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, 2)); qd = rng.normal(size=(n, 2)); tau = rng.normal(size=(n, 2))
    model.qdd(q, qd, tau)
    t = time.perf_counter()
    for _ in range(reps):
        model.qdd(q, qd, tau)
    return (time.perf_counter() - t) / reps * 1e6


# ---------------------------------------------------------------------------
def main():
    p = nd.load_params()
    pools, rows, norm = load_tests()
    q, qd, pool, masks = regions(rows)
    W = {s: windows(pools, s) for s in TEST_POOLS}
    U = load_U()
    info = json.loads((nd.RES_DIR / "p3_training.json").read_text())
    lam = info["lam_inv"]

    refs = {
        "analytical": nd.Analytical(p),
        "thin_rod": nd.Analytical(replace(p, I_c=p.m_p * (2 * p.l) ** 2 / 12.0)),
        "plus1pct": nd.Analytical(replace(p, m_c=p.m_c * 1.01, m_p=p.m_p * 1.01, l=p.l * 1.01,
                                          I_c=p.I_c * 1.01, b_x=p.b_x * 1.01, b_th=p.b_th * 1.01)),
    }
    res = dict(normaliser_qdd_std=norm.tolist(), regions={k: int(v.sum()) for k, v in masks.items()},
               n_windows={s: int(W[s]["q0"].shape[0]) for s in TEST_POOLS}, models={})
    heat = {}

    def evaluate(name, model, meta, sup=None):
        t0 = time.time()
        r = dict(meta)
        r["one_step"], e = one_step(model, rows, masks, norm, sup, q, qd)
        r["rollout"] = {s: rollout_metrics(model, W[s]) for s in TEST_POOLS}
        if isinstance(model, nd.Analytical):
            r["drift"] = drift(nd.Analytical(model.p, damping=False), U, p)
        elif isinstance(model, nd.DeLaN):
            model.damping_on = False
            r["drift"] = drift(model, U, p)
            model.damping_on = True
            r["drift_learned_D"] = drift(model, U, p)
            r["inverse"] = inverse_metrics(model, rows)
            r["damping"] = model.d.tolist()
        else:
            r["drift"] = drift(model, U, p)
        if sup is not None:
            dist = support_distance(sup, q, qd)
            en = np.sqrt((e ** 2).mean(1))
            r["error_vs_support_distance"] = {
                str(int(dd)): dict(n=int((dist == dd).sum()), nrmse=float(np.sqrt(np.mean(en[dist == dd] ** 2))))
                for dd in np.unique(np.minimum(dist, 20))}
        heat[name] = np.sqrt((e ** 2).mean(1))
        res["models"][name] = r
        print(f"{name:<22} {time.time() - t0:5.1f}s  1-step all "
              f"{r['one_step']['all']['nrmse_xdd']:.2e}/{r['one_step']['all']['nrmse_thdd']:.2e}  "
              f"VPT R {r['rollout']['R_test']['vpt_mean_s']:.2f}s", flush=True)

    for name, m in refs.items():
        evaluate(name, m, dict(kind="reference"))
    for r in info["runs"]:
        name = r["name"]
        model = nd.load_model(name)
        sup = support_hist(*(lambda d: (d["q"], d["qd"]))(nd.train_set(model.ck)))
        evaluate(name, model, dict(kind=r["kind"], regime=r["regime"], N=r["N"], seed=r["seed"],
                                   best_val=r["best_val"], early_stopped=r["stopped_step"] < 30000), sup)

    res["compute_us_per_qdd_batch512"] = {
        "analytical": timing(refs["analytical"]),
        "mlp": timing(nd.load_model("mlp_B20_s0")),
        "delan": timing(nd.load_model(nd.run_name("delan", "B", 20, 0, lam))),
    }
    res["summary"] = summarise(res)
    (nd.RES_DIR / "p4_eval.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(nd.RES_DIR / "p4_heat.npz", q=q, qd=qd, pool=pool,
                        **{k: v.astype(np.float32) for k, v in heat.items()})
    print(json.dumps(res["summary"], indent=1)[:6000])
    plots(res, q, qd, heat, info, lam)


# ---------------------------------------------------------------------------
def _agg(vals):
    v = np.asarray(vals, float)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), min=float(v.min()), max=float(v.max()),
                n=int(len(v)))


def summarise(res):
    S = {}
    runs = {k: v for k, v in res["models"].items() if v["kind"] in ("mlp", "delan")}
    cfgs = sorted({(v["regime"], v["N"]) for v in runs.values()})
    for reg, N in cfgs:
        for kind in ("mlp", "delan"):
            rs = [v for v in runs.values() if v["kind"] == kind and v["regime"] == reg and v["N"] == N]
            if not rs:
                continue
            row = {}
            for region in ("all", "hanging", "mid", "upright", "balance (S_test)", "OOR-angle (E_test)",
                           "OOR-velocity (E_test)", "in support", "out of support"):
                vals = [np.hypot(r["one_step"][region]["nrmse_xdd"], r["one_step"][region]["nrmse_thdd"]) / np.sqrt(2)
                        for r in rs if region in r["one_step"]]
                if vals:
                    row[f"one_step_nrmse::{region}"] = _agg(vals)
            for pool in TEST_POOLS:
                for hz in HORIZONS:
                    row[f"rollout_theta::{pool}::{hz}"] = _agg([r["rollout"][pool][hz]["theta_median"] for r in rs])
                row[f"vpt::{pool}"] = _agg([r["rollout"][pool]["vpt_mean_s"] for r in rs])
            for g in ("small", "large", "rotating", "fast_rotating"):
                row[f"drift::{g}"] = _agg([r["drift"][g]["dH_true_max"] for r in rs])
            S[f"{kind}_{reg}{N}"] = row
    for name in ("analytical", "thin_rod", "plus1pct"):
        r = res["models"][name]
        S[name] = {f"rollout_theta::{pool}::{hz}": r["rollout"][pool][hz]["theta_median"]
                   for pool in TEST_POOLS for hz in HORIZONS}
        S[name].update({f"vpt::{pool}": r["rollout"][pool]["vpt_mean_s"] for pool in TEST_POOLS})
        S[name].update({f"drift::{g}": r["drift"][g]["dH_true_max"] for g in ("small", "large", "rotating", "fast_rotating")})
    return S


# ---------------------------------------------------------------------------
def plots(res, q, qd, heat, info, lam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    # same colours as cartpole.plotting.MODEL_COLORS (analytical drawn black as the reference)
    C = {"mlp": "#d95f02", "delan": "#7570b3", "analytical": "k", "thin_rod": "#66a61e", "plus1pct": "#999999"}
    S = res["summary"]

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- data efficiency (regime A) -------------------------------------------
    Ns = [1, 5, 20]
    panels = [("one_step_nrmse::all", "one-step nRMSE, all test rows"),
              ("rollout_theta::E_test::1s", r"rollout RMSE of $\theta$ at 1 s, E_test [rad]"),
              ("vpt::S_test", "valid-prediction time, S_test (upright) [s]"),
              ("drift::large", r"energy drift, 'large' group, 10 s [$m_pgl$]")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for kind in ("mlp", "delan"):
            med = [S[f"{kind}_A{N}"][key]["median"] for N in Ns]
            lo = [S[f"{kind}_A{N}"][key]["min"] for N in Ns]
            hi = [S[f"{kind}_A{N}"][key]["max"] for N in Ns]
            ax.plot(Ns, med, "o-", color=C[kind], lw=2, label="MLP" if kind == "mlp" else "DeLaN")
            ax.fill_between(Ns, lo, hi, color=C[kind], alpha=0.18, lw=0)
        if not key.startswith("one_step"):
            ref = S["analytical"].get(key)
            # exact model: rollout error is round-off and drift is the RK4 floor; draw it only if on scale
            if ref is not None and ref > 1e-6:
                ax.axhline(ref, color="k", ls="--", lw=1, label="analytical")
        ax.set_xscale("log")
        if not key.startswith("vpt"):
            ax.set_yscale("log")
        ax.set_xticks(Ns); ax.set_xticklabels(Ns)
        ax.set(xlabel="training episodes N (regime A)", title=title)
    axes[0].legend()
    fig.suptitle("Data efficiency, regime A (median over seeds; band = min-max)", y=1.02)
    save(fig, "p4_data_efficiency")

    # --- rollout error vs horizon, per pool, N=20 -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    hz_s = [0.5, 1, 2]
    for ax, pool in zip(axes, TEST_POOLS):
        # the exact analytical model is round-off (<= 5e-15 rad, see results) and is left out of the log plot
        for name, st in (("thin_rod", "-"), ("plus1pct", "-")):
            ax.plot(hz_s, [S[name][f"rollout_theta::{pool}::{h}"] for h in HORIZONS], st, color=C[name],
                    marker="s", lw=1.2, label={"analytical": "analytical (exact)", "thin_rod": "analytical, thin rod",
                                               "plus1pct": "analytical, +1 % params"}[name])
        for kind in ("mlp", "delan"):
            for reg, ls in (("A", ":"), ("B", "-")):
                k = f"{kind}_{reg}20"
                ax.plot(hz_s, [S[k][f"rollout_theta::{pool}::{h}"]["median"] for h in HORIZONS], ls, marker="o",
                        color=C[kind], lw=2, label=f"{'MLP' if kind == 'mlp' else 'DeLaN'} {reg}20")
        ax.set_yscale("log"); ax.set(title=pool, xlabel="horizon [s]")
        ax.set_xticks(hz_s)
    axes[0].set_ylabel(r"median RMSE of wrapped $\theta$ [rad]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Open-loop rollout error (replayed recorded inputs; median over windows, median over seeds; exact analytical model at round-off, not shown)", y=1.02)
    save(fig, "p4_rollout")

    # --- energy drift per group ---------------------------------------------------
    groups = ["small", "large", "rotating", "fast_rotating"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    entries = [("analytical", "analytical (RK4 floor)", C["analytical"])] + \
              [(f"{k}_{r}20", f"{'MLP' if k == 'mlp' else 'DeLaN'} {r}20", C[k]) for r in ("A", "B") for k in ("mlp", "delan")]
    w = 0.16
    for i, (key, lab, col) in enumerate(entries):
        vals = [S[key][f"drift::{g}"] if key == "analytical" else S[key][f"drift::{g}"]["median"] for g in groups]
        hatch = "//" if key.endswith("A20") else None
        ax.bar(np.arange(4) + (i - 2) * w, vals, w, color=col, hatch=hatch, label=lab, alpha=0.85 if "B20" in key else 0.6)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(groups)
    ax.set_ylabel(r"max$_t\,|H_{true}(\hat s_t)-H_{true}(s_0)|/(m_pgl)$, worst of group, median over seeds")
    ax.set_title("Energy drift over 10 s, unforced and undamped (DeLaN with D = 0)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "p4_energy_drift")

    # --- OOR heat maps ---------------------------------------------------------
    def best(kind, reg, N):
        rs = [r for r in info["runs"] if r["kind"] == kind and r["regime"] == reg and r["N"] == N]
        return min(rs, key=lambda r: r["best_val"])["name"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    th = nd.wrap(q[:, 1]); thd = qd[:, 1]
    for row, reg in enumerate(("A", "B")):
        for col, kind in enumerate(("mlp", "delan")):
            name = best(kind, reg, 20)
            ax = axes[row, col]
            Hs, xe, ye = np.histogram2d(th, thd, bins=[np.linspace(-np.pi, np.pi, 64), np.linspace(-12, 12, 49)],
                                        weights=heat[name] ** 2)
            Hc, _, _ = np.histogram2d(th, thd, bins=[xe, ye])
            with np.errstate(all="ignore"):
                Z = np.sqrt(Hs / Hc)
            im = ax.pcolormesh(xe, ye, np.log10(Z).T, vmin=-4, vmax=1, cmap="magma")
            ck = nd.load_checkpoint(nd.WEIGHT_DIR / f"{name}.pt")
            tr = nd.train_set(ck)
            sup = support_hist(tr["q"], tr["qd"]).astype(float)
            ax.contour(0.5 * (_ETH[1:] + _ETH[:-1]), 0.5 * (_ETHD[1:] + _ETHD[:-1]), sup.T, levels=[0.5],
                       colors="cyan", linewidths=1.2)
            ax.set_title(f"{'MLP' if kind == 'mlp' else 'DeLaN'} {reg}20 ({name.split('_')[2]})")
    for ax in axes[1]:
        ax.set_xlabel(r"$\theta$ [rad] (0 = upright)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    fig.colorbar(im, ax=axes, label=r"log$_{10}$ one-step nRMSE")
    fig.suptitle("One-step error over the test states (cyan: density support of the training set)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"p4_oor_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- error vs distance to support -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind in ("mlp", "delan"):
        for reg, ls in (("A", ":"), ("B", "-")):
            rs = [v for v in res["models"].values() if v.get("kind") == kind and v.get("regime") == reg and v.get("N") == 20]
            ds = sorted({int(d) for r in rs for d in r["error_vs_support_distance"]})
            med = [np.median([r["error_vs_support_distance"][str(d)]["nrmse"] for r in rs
                              if str(d) in r["error_vs_support_distance"]]) for d in ds]
            ax.plot(ds, med, ls, marker="o", ms=3, color=C[kind], lw=2, label=f"{'MLP' if kind == 'mlp' else 'DeLaN'} {reg}20")
    ax.set_yscale("log")
    ax.set(xlabel="Chebyshev distance to nearest supported bin (bins; 20 = 20 or more)",
           ylabel="one-step nRMSE (median over seeds)", title="Extrapolation: error vs distance from the training data")
    ax.legend()
    save(fig, "p4_error_vs_support_distance")


def plots_only():
    res = json.loads((nd.RES_DIR / "p4_eval.json").read_text())
    z = np.load(nd.RES_DIR / "p4_heat.npz")
    heat = {k: z[k].astype(np.float64) for k in z.files if k not in ("q", "qd", "pool")}
    info = json.loads((nd.RES_DIR / "p3_training.json").read_text())
    plots(res, z["q"], z["qd"], heat, info, info["lam_inv"])


if __name__ == "__main__":
    plots_only() if "--plots-only" in sys.argv else main()
