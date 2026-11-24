"""Phase 4: open-loop evaluation of every model (PLAN Phase 4).

Models: analytical (oracle), thin-rod, analytical +1 % on every parameter, and every
trained MLP / DeLaN of the Phase 3 grid (all regimes, N, seeds). All evaluation is float64
and goes through the same RK4 (`cartpole.integrators.rk4_step`).

Stages (each writes results/p4_<stage>.json):
  onestep  q_ddot nRMSE and one-step state error per pool and region (hanging / mid / upright /
           balance / OOR-angle / OOR-velocity / density-supported / unsupported)
  rollout  open-loop replay of the recorded u: wrapped-theta and x RMSE at 0.5 / 1 / 2 s
           (median, IQR over windows), valid-prediction time, error by start region
  energy   10 s unforced, undamped drift on the U pool, per group, with support fraction
  oor      error heat map over (theta, thetadot) and error vs distance to the training support
  misc     DeLaN inverse dynamics, identified parameters (from Phase 3), compute cost
  plot     figures/p4_*.{png,pdf} and results/p4_tables.md (needs the JSON of the stages above)

Usage:  python scripts/p4_eval.py [--stage all|onestep|rollout|energy|oor|misc|plot] [--runs all|best]
        --runs best evaluates only the lowest-validation-loss seed of each (model, regime, N).
Outputs: results/p4_*.json, results/p4_oor_maps.npz, results/p4_tables.md, figures/p4_*
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole.analytical import AnalyticalModel  # noqa: E402
from cartpole.data import SUPPORT_MIN_COUNT, in_support, load_episodes, support_histogram  # noqa: E402
from cartpole.dataset import build  # noqa: E402
from cartpole.integrators import rk4_step, rollout  # noqa: E402
from cartpole.loading import ANALYTIC, best_run, get_params, grid_runs, load_any  # noqa: E402
from cartpole.plotting import DATA_DIR, MODEL_COLORS, MODEL_LABELS, RES_DIR, savefig, setup_style  # noqa: E402

torch.set_default_dtype(torch.float64)

POOLS = ("R_test", "E_test", "S_test")
HORIZONS = {"0.5s": 50, "1s": 100, "2s": 200}
STRIDE = 25
GROUPS = ("small", "large", "rotating", "fast_rotating")


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def clean(o):
    """Make an object JSON-safe: numpy -> python, non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_json(name: str, obj):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"{name}.json").write_text(json.dumps(clean(obj), indent=1))
    print(f"wrote results/{name}.json", flush=True)


def load_json(name: str):
    return json.loads((RES_DIR / f"{name}.json").read_text())


def nz(v):
    return np.nan if v is None else v


_POOL: dict = {}


def pool(name: str):
    if name not in _POOL:
        _POOL[name] = load_episodes(DATA_DIR / f"{name}.npz")
    return _POOL[name]


_SUP: dict = {}


def train_support(run: dict):
    """(q, qd) rows of the run's training set, for the density-based support."""
    if run["name"] not in _SUP:
        d = build(run["regime"], run["N"], run["seed"], dtype=torch.float64)["train"]
        _SUP[run["name"]] = (d["q"].numpy(), d["qd"].numpy())
    return _SUP[run["name"]]


def thd_max_r() -> float:
    """Largest |thetadot| in R_train (PART2 section 6): the OOR-velocity threshold."""
    return float(json.loads((RES_DIR / "p2_dataset.json").read_text())["splits"]["R_train"]["max_abs_thd"])


def region_masks(q, qd, thd_max):
    th = wrap(q[:, 1])
    c, w = np.cos(th), qd[:, 1]
    return {
        "all": np.ones(len(c), bool),
        "hanging": c < -0.8,
        "mid": (c >= -0.8) & (c <= 0.8),
        "upright": c > 0.8,
        "balance": (np.abs(th) < 0.3) & (np.abs(w) < 2.0),
        "oor_angle": c > 0.0,
        "oor_velocity": (c < -0.8) & (np.abs(w) > thd_max),
    }


def model_list(mode: str):
    """[(name, spec, run-or-None)]. Analytic variants first, then the Phase 3 grid."""
    models = [(n, n, None) for n in ANALYTIC]
    runs = grid_runs()
    if mode == "best":
        keep = {best_run(k, r, N)["name"] for (k, r, N) in {(x["kind"], x["regime"], x["N"]) for x in runs}}
        runs = [r for r in runs if r["name"] in keep]
    models += [(r["name"], r["path"], r) for r in runs]
    return models


# ---------------------------------------------------------------------------------------
# one-step
# ---------------------------------------------------------------------------------------
def onestep_arrays(eps):
    """Rows (state, force, label, next state) with the rail limit inactive at t and t+1."""
    keys = ("q", "qd", "tau", "qdd", "q1", "qd1")
    out = {k: [] for k in keys}
    for e in eps:
        lim = e["limit_active"].astype(bool)
        keep = ~(lim | np.concatenate([lim[1:], [False]]))
        q1 = np.concatenate([e["qpos"][1:], e["qpos_final"][None]])
        qd1 = np.concatenate([e["qvel"][1:], e["qvel_final"][None]])
        for k, v in zip(keys, (e["qpos"], e["qvel"], e["qfrc_actuator"] + e["qfrc_constraint"], e["qacc"], q1, qd1)):
            out[k].append(v[keep])
    return {k: np.concatenate(v) for k, v in out.items()}


def predict(model, d, h, chunk=20000):
    q, qd, tau = (torch.as_tensor(d[k]) for k in ("q", "qd", "tau"))
    a, b, c = [], [], []
    with torch.no_grad():
        for i in range(0, len(q), chunk):
            s = slice(i, i + chunk)
            a.append(model.qdd(q[s], qd[s], tau[s]))
            q1, qd1 = rk4_step(model, q[s], qd[s], tau[s], h)
            b.append(q1)
            c.append(qd1)
    return [torch.cat(x).numpy() for x in (a, b, c)]


def stage_onestep(models, p):
    thd_max = thd_max_r()
    pools = {n: onestep_arrays(pool(n)[0]) for n in POOLS}
    sig = {n: d["qdd"].std(0) for n, d in pools.items()}
    out = {}
    for name, spec, run in models:
        t0 = time.time()
        model = load_any(spec)
        sup = train_support(run) if run else None
        res = {}
        for pn, d in pools.items():
            qdd_hat, q1h, qd1h = predict(model, d, p.dt)
            e = qdd_hat - d["qdd"]
            dq = q1h - d["q1"]
            dq[:, 1] = wrap(dq[:, 1])
            dqd = qd1h - d["qd1"]
            masks = region_masks(d["q"], d["qd"], thd_max)
            if sup is not None:
                s = in_support(sup[0], sup[1], d["q"], d["qd"])
                masks["supported"], masks["unsupported"] = s, ~s
            res[pn] = {}
            for rn, m in masks.items():
                if m.sum() == 0:
                    continue
                rms = lambda a: np.sqrt(np.mean(a[m] ** 2, 0))          # noqa: E731
                nr = rms(e) / sig[pn]
                sq, sv = rms(dq), rms(dqd)
                res[pn][rn] = dict(n=int(m.sum()), nrmse_xdd=nr[0], nrmse_thdd=nr[1],
                                   state_rmse=dict(x=sq[0], theta=sq[1], xd=sv[0], thd=sv[1]))
        out[name] = res
        print(f"onestep {name:<22} {time.time() - t0:5.1f}s  R_test thdd nRMSE "
              f"{out[name]['R_test']['all']['nrmse_thdd']:.3e}", flush=True)
    save_json("p4_onestep", out)


# ---------------------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------------------
def make_windows(eps, p, T=200, stride=STRIDE):
    """Windows of T steps starting every `stride` rows; windows that touch the rail limit are dropped."""
    q0, qd0, tau, qt, qdt = [], [], [], [], []
    for e in eps:
        n = len(e["u"])
        QT = np.concatenate([e["qpos"], e["qpos_final"][None]])
        QDT = np.concatenate([e["qvel"], e["qvel_final"][None]])
        lim = e["limit_active"].astype(bool)
        for s in range(0, n - T + 1, stride):
            if lim[s:s + T].any():
                continue
            q0.append(QT[s])
            qd0.append(QDT[s])
            tau.append(np.stack([p.gear * e["u"][s:s + T], np.zeros(T)], 1))
            qt.append(QT[s:s + T + 1])
            qdt.append(QDT[s:s + T + 1])
    if not q0:
        return None
    return dict(q0=np.array(q0), qd0=np.array(qd0), tau=np.stack(tau, 1), qt=np.stack(qt, 1), qdt=np.stack(qdt, 1))


def iqr(a):
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def rollout_metrics(model, W, p, thd_max, T=200):
    with torch.no_grad():
        Q, QD = rollout(model, torch.as_tensor(W["q0"]), torch.as_tensor(W["qd0"]), torch.as_tensor(W["tau"]), p.dt)
    Q = Q.numpy()
    dth = np.nan_to_num(np.abs(wrap(Q[..., 1] - W["qt"][..., 1])), nan=np.inf, posinf=np.inf)    # (T+1, W)
    dx = np.nan_to_num(np.abs(Q[..., 0] - W["qt"][..., 0]), nan=np.inf, posinf=np.inf)
    out = {}
    rth_by_h = {}
    for hn, hs in HORIZONS.items():
        rth = np.sqrt(np.mean(dth[:hs + 1] ** 2, 0))          # per-window RMS over the first hs steps
        rx = np.sqrt(np.mean(dx[:hs + 1] ** 2, 0))
        rth_by_h[hn] = rth
        out[hn] = dict(theta_median=np.median(rth), theta_iqr=iqr(rth), x_median=np.median(rx), x_iqr=iqr(rx))
    bad = dth > 0.1
    first = np.where(bad.any(0), bad.argmax(0), T)
    out["valid_time_s"] = float(first.mean() * p.dt)
    out["n_windows"] = int(dth.shape[1])
    out["theta_err_curve"] = np.median(dth, 1)
    masks = region_masks(W["q0"], W["qd0"], thd_max)
    out["by_start_region_0.5s_theta_median"] = {
        rn: float(np.median(rth_by_h["0.5s"][m])) for rn, m in masks.items() if m.sum() > 0}
    out["by_start_region_n"] = {rn: int(m.sum()) for rn, m in masks.items() if m.sum() > 0}
    return out


def stage_rollout(models, p):
    thd_max = thd_max_r()
    windows = {n: make_windows(pool(n)[0], p) for n in POOLS}
    out = {}
    for name, spec, _ in models:
        t0 = time.time()
        model = load_any(spec)
        out[name] = {pn: rollout_metrics(model, W, p, thd_max) for pn, W in windows.items() if W is not None}
        r = out[name]["R_test"]["1s"]["theta_median"]
        print(f"rollout {name:<22} {time.time() - t0:5.1f}s  R_test 1 s theta RMSE median {r:.3e}", flush=True)
    save_json("p4_rollout", out)


# ---------------------------------------------------------------------------------------
# energy drift
# ---------------------------------------------------------------------------------------
def stage_energy(models, p):
    eps, meta = pool("U")
    groups = np.array([str(g) for g in meta["groups"]])
    truth = AnalyticalModel(p, damping=False)
    agl = p.mpl * p.g
    q0 = torch.as_tensor(np.stack([e["qpos"][0] for e in eps]))
    qd0 = torch.as_tensor(np.stack([e["qvel"][0] for e in eps]))
    T = min(len(e["u"]) for e in eps)
    QT = np.stack([np.concatenate([e["qpos"][:T], e["qpos_final"][None]]) for e in eps], 1)     # (T+1, E, 2)
    QDT = np.stack([np.concatenate([e["qvel"][:T], e["qvel_final"][None]]) for e in eps], 1)
    tau = torch.zeros(T, len(eps), 2)

    def summarise(Q, QD, sup, own_H=None):
        Q, QD = torch.as_tensor(Q), torch.as_tensor(QD)
        H = truth.energy(Q, QD)
        drift = np.nan_to_num(((H - H[0]).abs().max(0).values / agl).numpy(), nan=np.inf)
        px = truth.momentum_x(Q, QD)
        pdrift = np.nan_to_num((px - px[0]).abs().max(0).values.numpy(), nan=np.inf)
        Qn = Q.numpy()
        eth = np.nan_to_num(np.abs(wrap(Qn[..., 1] - QT[..., 1])).max(0), nan=np.inf)
        res = {}
        for g in GROUPS:
            m = groups == g
            if m.sum() == 0:
                continue
            entry = dict(drift_median=np.median(drift[m]), drift_max=drift[m].max(),
                         px_drift_median=np.median(pdrift[m]), max_theta_err_median=np.median(eth[m]),
                         drift_per_episode=drift[m])
            if own_H is not None:
                entry["own_H_drift_median"] = np.median(own_H[m])
            if sup is not None:
                rows_q = QT[:, m].reshape(-1, 2)
                rows_qd = QDT[:, m].reshape(-1, 2)
                entry["support_fraction"] = float(in_support(sup[0], sup[1], rows_q, rows_qd).mean())
            res[g] = entry
        return res

    out = {"mujoco": summarise(QT, QDT, None)}
    for name, spec, run in models:
        t0 = time.time()
        model = load_any(spec)
        sup = train_support(run) if run else None
        variants = [(name, None)]
        if run is not None and run["kind"] == "delan":
            variants = [(f"{name}|D0", False), (f"{name}|Dlearned", True)]
        for label, damp_on in variants:
            if isinstance(model, AnalyticalModel):
                model.damping = False            # the U pool is undamped
            if damp_on is not None:
                model.damping_on = damp_on
            with torch.no_grad():
                Q, QD = rollout(model, q0, qd0, tau, p.dt)
                own = None
                if run is not None and run["kind"] == "delan":
                    Hm = model.energy(Q, QD)
                    own = np.nan_to_num(((Hm - Hm[0]).abs().max(0).values / agl).numpy(), nan=np.inf)
            out[label] = summarise(Q.numpy(), QD.numpy(), sup, own)
        print(f"energy  {name:<22} {time.time() - t0:5.1f}s", flush=True)
    save_json("p4_energy", out)


# ---------------------------------------------------------------------------------------
# OOR heat map and distance to support
# ---------------------------------------------------------------------------------------
def stage_oor(p):
    from scipy.ndimage import distance_transform_cdt

    thd_max = thd_max_r()
    per_pool = [onestep_arrays(pool(n)[0]) for n in POOLS]
    d = {k: np.concatenate([a[k] for a in per_pool]) for k in ("q", "qd", "tau", "qdd")}
    sig = d["qdd"].std(0)
    names = ["analytical"]
    for kind in ("mlp", "delan"):
        for regime in ("A", "B"):
            names.append(best_run(kind, regime, 20))
    th, w = wrap(d["q"][:, 1]), d["qd"][:, 1]
    dummy_sup = support_histogram(d["q"][:1], d["qd"][:1])
    eth, ethd = dummy_sup[1], dummy_sup[2]
    nth, nthd = len(eth) - 1, len(ethd) - 1
    ii = np.clip(np.digitize(th, eth) - 1, 0, nth - 1)
    jj = np.clip(np.digitize(w, ethd) - 1, 0, nthd - 1)
    maps, curves = {"eth": eth, "ethd": ethd}, {}
    dd = dict(q=d["q"], qd=d["qd"], tau=d["tau"])
    for item in names:
        name, spec, run = ("analytical", "analytical", None) if item == "analytical" else (item["name"], item["path"], item)
        model = load_any(spec)
        qdd_hat, _, _ = predict(model, dd, p.dt)
        err = np.sqrt((((qdd_hat - d["qdd"]) / sig) ** 2).sum(1))
        sm = np.histogram2d(th, w, bins=[eth, ethd], weights=err)[0]
        cnt = np.histogram2d(th, w, bins=[eth, ethd])[0]
        maps[f"{name}__mean"] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        if run is not None:
            sq, sqd = train_support(run)
            sup = support_histogram(sq, sqd)[0] >= SUPPORT_MIN_COUNT
            maps[f"{name}__support"] = sup
            if sup.any():
                big = np.concatenate([sup, sup, sup], 0)          # wrap theta
                dm = distance_transform_cdt(~big, metric="chessboard")[nth:2 * nth]
                dist = dm[ii, jj]
                kmax = 10
                dist = np.minimum(dist, kmax)
                curves[name] = dict(
                    dist_bins=list(range(kmax + 1)),
                    mean_err=[float(err[dist == k].mean()) if (dist == k).any() else None for k in range(kmax + 1)],
                    n=[int((dist == k).sum()) for k in range(kmax + 1)])
        print(f"oor     {name}", flush=True)
    np.savez_compressed(RES_DIR / "p4_oor_maps.npz", **maps)
    save_json("p4_oor", dict(models=[n if isinstance(n, str) else n["name"] for n in names], distance_curves=curves,
                             bin_th=float(eth[1] - eth[0]), bin_thd=float(ethd[1] - ethd[0]),
                             note="error = sqrt(sum_j ((qdd_hat_j - qdd_j)/sigma_j)^2), one step, pooled R/E/S test rows"))


# ---------------------------------------------------------------------------------------
# misc: inverse dynamics, identification, compute cost
# ---------------------------------------------------------------------------------------
def stage_misc(models, p):
    out = {"inverse_dynamics": {}, "compute_us_per_batch512": {}}
    pools = {n: onestep_arrays(pool(n)[0]) for n in POOLS}
    for name, spec, run in models:
        if run is None or run["kind"] != "delan":
            continue
        model = load_any(spec)
        res = {}
        for pn, d in pools.items():
            q, qd, qdd = (torch.as_tensor(d[k]) for k in ("q", "qd", "qdd"))
            with torch.no_grad():
                th_ = torch.cat([model.inverse(q[i:i + 20000], qd[i:i + 20000], qdd[i:i + 20000])[0]
                                 for i in range(0, len(q), 20000)]).numpy()
            tau = d["tau"]
            res[pn] = dict(
                nrmse_tau_x=float(np.sqrt(np.mean((th_[:, 0] - tau[:, 0]) ** 2)) / tau[:, 0].std()),
                rms_tau_theta_row=float(np.sqrt(np.mean(th_[:, 1] ** 2))),
                tau_theta_row_rel=float(np.sqrt(np.mean(th_[:, 1] ** 2)) / (p.J * d["qdd"][:, 1].std())))
        out["inverse_dynamics"][name] = res
        print(f"inverse {name}", flush=True)
    # compute cost: analytical, best MLP-B20, best DeLaN-B20
    rng = np.random.default_rng(0)
    q = torch.as_tensor(rng.normal(size=(512, 2)))
    qd = torch.as_tensor(2.0 * rng.normal(size=(512, 2)))
    tau = torch.as_tensor(5.0 * rng.normal(size=(512, 2)))
    for label, spec in (("analytical", "analytical"), ("mlp", best_run("mlp", "B", 20)["path"]),
                        ("delan", best_run("delan", "B", 20)["path"])):
        model = load_any(spec)
        with torch.no_grad():
            for _ in range(30):
                model.qdd(q, qd, tau)
            ts = []
            for _ in range(300):
                t0 = time.perf_counter()
                model.qdd(q, qd, tau)
                ts.append(time.perf_counter() - t0)
        out["compute_us_per_batch512"][label] = float(np.median(ts) * 1e6)
    ident = RES_DIR / "p3_identification.json"
    if ident.exists():
        out["identification_summary"] = json.loads(ident.read_text())["summary_rel_error_on_training_angles"]
    else:
        out["identification_summary"] = None
        print("note: results/p3_identification.json not found (run scripts/p3_structure.py)")
    save_json("p4_misc", out)


# ---------------------------------------------------------------------------------------
# plots and tables
# ---------------------------------------------------------------------------------------
def run_index():
    return {r["name"]: r for r in grid_runs()}


def style_of(name, runs):
    if name in ANALYTIC:
        return dict(color={"analytical": MODEL_COLORS["analytical"], "thin_rod": MODEL_COLORS["thin_rod"],
                           "analytical_p1": "#888888"}[name],
                    ls={"analytical": "-", "thin_rod": "-.", "analytical_p1": ":"}[name], lw=1.6)
    r = runs[name.split("|")[0]]
    return dict(color=MODEL_COLORS[r["kind"]], ls="--" if r["regime"] == "A" else "-", lw=1.8)


def label_of(name, runs):
    if name in ANALYTIC:
        return {"analytical": "Analytical", "thin_rod": "Analytical (thin rod)", "analytical_p1": "Analytical (+1 %)"}[name]
    base = name.split("|")[0]
    r = runs[base]
    s = f"{'MLP' if r['kind'] == 'mlp' else 'DeLaN'} {r['regime']}{r['N']} (s{r['seed']})"
    if "|D0" in name:
        s += ", D=0"
    if "|Dlearned" in name:
        s += ", learned D"
    return s


def selected(runs):
    sel = list(ANALYTIC)
    for kind in ("mlp", "delan"):
        for regime in ("A", "B"):
            sel.append(best_run(kind, regime, 20)["name"])
    return sel


def plot_rollout(roll, runs, p):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, pn in zip(axes, POOLS):
        for name in selected(runs):
            if name in roll and pn in roll[name]:
                c = np.array([nz(v) for v in roll[name][pn]["theta_err_curve"]])
                t = np.arange(len(c)) * p.dt
                ax.semilogy(t, np.maximum(c, 1e-12), label=label_of(name, runs), **style_of(name, runs))
        ax.set(title=f"{pn}: median |wrapped theta error|", xlabel="rollout time [s]")
    axes[0].set_ylabel("rad")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p4_rollout")


def plot_energy(en, runs):
    import matplotlib.pyplot as plt
    names = ["mujoco", "analytical"]
    for kind in ("mlp", "delan"):
        for regime in ("A", "B"):
            b = best_run(kind, regime, 20)["name"]
            names += [b] if kind == "mlp" else [f"{b}|D0", f"{b}|Dlearned"]
    names = [n for n in names if n in en]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [nz(en[n].get(g, {}).get("drift_median")) for g in GROUPS]
        st = style_of(n, runs) if n != "mujoco" else dict(color=MODEL_COLORS["mujoco"])
        ax.bar(np.arange(len(GROUPS)) + i * width, np.maximum(vals, 1e-12), width, color=st["color"],
               hatch="//" if "|Dlearned" in n else "",
               alpha=0.55 if runs.get(n.split("|")[0], {}).get("regime") == "A" else 0.9,
               label="MuJoCo (floor)" if n == "mujoco" else label_of(n, runs))
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(GROUPS)) + 0.4)
    ax.set_xticklabels(GROUPS)
    ax.set(ylabel=r"median $\max_t |H_{true}-H_0| / (m_p g l)$", title="Energy drift, 10 s, unforced and undamped")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig(fig, "p4_energy")


def aggregate(values: dict, runs):
    """values: model name -> scalar. Returns {(kind, regime, N): [values over seeds]}."""
    agg = {}
    for name, v in values.items():
        r = runs.get(name.split("|")[0])
        if r is None or v is None:
            continue
        agg.setdefault((r["kind"], r["regime"], r["N"]), []).append(v)
    return agg


def plot_data_efficiency(one, roll, en, runs):
    import matplotlib.pyplot as plt
    metrics = [
        ("one-step nRMSE of $\\ddot\\theta$ (R_test)",
         {n: v["R_test"]["all"]["nrmse_thdd"] for n, v in one.items() if "R_test" in v}),
        ("1 s rollout, median wrapped-$\\theta$ RMSE [rad] (R_test)",
         {n: v["R_test"]["1s"]["theta_median"] for n, v in roll.items() if "R_test" in v}),
        ("energy drift, group 'small' (median, $/m_pgl$)",
         {n: v["small"]["drift_median"] for n, v in en.items() if "small" in v and "|D0" not in n}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (title, vals) in zip(axes, metrics):
        vals = {n: nz(v) for n, v in vals.items()}
        agg = aggregate({k.replace("|Dlearned", ""): v for k, v in vals.items()}, runs)
        for kind in ("mlp", "delan"):
            Ns = sorted(N for (k, r, N) in agg if k == kind and r == "A")
            m = [np.mean(agg[(kind, "A", N)]) for N in Ns]
            s = [np.std(agg[(kind, "A", N)]) for N in Ns]
            ax.errorbar(Ns, m, yerr=s, marker="o", capsize=3, color=MODEL_COLORS[kind],
                        label="MLP" if kind == "mlp" else "DeLaN")
        if "analytical" in vals:
            ax.axhline(vals["analytical"], color=MODEL_COLORS["analytical"], ls="-", label="Analytical")
        ax.set(xscale="log", yscale="log", xlabel="training episodes $N$ (regime A)", title=title)
        ax.set_xticks([1, 5, 20])
        ax.set_xticklabels(["1", "5", "20"])
    axes[0].legend()
    fig.suptitle("Data efficiency (mean $\\pm$ std over seeds)", y=1.02)
    fig.tight_layout()
    savefig(fig, "p4_data_efficiency")


def plot_oor():
    import matplotlib.pyplot as plt
    z = np.load(RES_DIR / "p4_oor_maps.npz")
    info = load_json("p4_oor")
    eth, ethd = z["eth"], z["ethd"]
    names = info["models"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), sharey=True)
    vals = np.concatenate([z[f"{n}__mean"].ravel() for n in names])
    vmax = np.nanpercentile(vals, 99)
    vmin = max(np.nanpercentile(vals[vals > 0], 1), 1e-8)
    runs = run_index()
    from matplotlib.colors import LogNorm
    for ax, n in zip(np.atleast_1d(axes), names):
        im = ax.pcolormesh(eth, ethd, z[f"{n}__mean"].T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="magma", shading="flat")
        if f"{n}__support" in z.files:
            ax.contour(0.5 * (eth[:-1] + eth[1:]), 0.5 * (ethd[:-1] + ethd[1:]), z[f"{n}__support"].T.astype(float),
                       levels=[0.5], colors="cyan", linewidths=1.2)
        ax.set(title=label_of(n, runs) if n != "analytical" else "Analytical", xlabel=r"$\theta$ [rad]")
        ax.set_xlim(-np.pi, np.pi)
    np.atleast_1d(axes)[0].set_ylabel(r"$\dot\theta$ [rad/s]")
    fig.colorbar(im, ax=axes, label="one-step normalised $\\ddot q$ error (cyan: training support)")
    savefig(fig, "p4_oor_heatmap")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for n, c in info["distance_curves"].items():
        y = [nz(v) for v in c["mean_err"]]
        ax.semilogy(c["dist_bins"], y, marker="o", label=label_of(n, runs), **style_of(n, runs))
    ax.set(xlabel="distance to training support [bins, Chebyshev]", ylabel="mean one-step normalised $\\ddot q$ error",
           title="Error vs distance from the density-based support")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "p4_oor_distance")


def fmt(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.2e}"


def table_by_config(values: dict, runs, title: str):
    agg = aggregate(values, runs)
    lines = [f"**{title}** (median [min, max] over seeds)", "", "| config | MLP | DeLaN |", "|---|---|---|"]
    configs = sorted({(r, N) for (_, r, N) in agg}, key=lambda t: (t[0], t[1]))
    for r, N in configs:
        cells = []
        for kind in ("mlp", "delan"):
            v = agg.get((kind, r, N))
            cells.append("-" if not v else f"{fmt(np.median(v))} [{fmt(min(v))}, {fmt(max(v))}]")
        lines.append(f"| {r}{N} ({len(agg.get(('mlp', r, N), []))} seeds) | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def write_tables(one, roll, en, misc, runs):
    out = ["# Phase 4 tables (generated by scripts/p4_eval.py; paste into docs/RESULTS.md)\n"]
    for nm in ANALYTIC:
        if nm in one:
            out.append(f"- {nm}: R_test one-step nRMSE (xdd, thdd) = {fmt(one[nm]['R_test']['all']['nrmse_xdd'])}, "
                       f"{fmt(one[nm]['R_test']['all']['nrmse_thdd'])}; 1 s rollout theta RMSE median "
                       f"{fmt(roll[nm]['R_test']['1s']['theta_median']) if nm in roll else 'n/a'}")
    out.append("")
    for pn in POOLS:
        out.append(table_by_config({n: v[pn]["all"]["nrmse_thdd"] for n, v in one.items() if pn in v}, runs,
                                   f"One-step nRMSE of thdd on {pn}, all rows"))
        out.append("")
    for rn in ("upright", "balance", "oor_angle", "oor_velocity", "unsupported"):
        vals = {n: v["E_test"][rn]["nrmse_thdd"] for n, v in one.items() if "E_test" in v and rn in v["E_test"]}
        if vals:
            out.append(table_by_config(vals, runs, f"One-step nRMSE of thdd on E_test, region {rn}"))
            out.append("")
    for pn in POOLS:
        for hn in HORIZONS:
            vals = {n: v[pn][hn]["theta_median"] for n, v in roll.items() if pn in v}
            out.append(table_by_config(vals, runs, f"Rollout {hn}, median wrapped-theta RMSE [rad] on {pn}"))
            out.append("")
    for g in GROUPS:
        vals = {n: v[g]["drift_median"] for n, v in en.items() if g in v and "|D0" not in n}
        vals = {k.replace("|Dlearned", ""): v for k, v in vals.items()}
        out.append(table_by_config(vals, runs, f"Energy drift, group {g} (learned D for DeLaN), units of m_p g l"))
        out.append("")
    if misc:
        out.append("**Compute, microseconds per batched qdd evaluation (batch 512, float64, no_grad):** "
                   + ", ".join(f"{k} {v:.0f}" for k, v in misc["compute_us_per_batch512"].items()))
    (RES_DIR / "p4_tables.md").write_text("\n".join(out))
    print("wrote results/p4_tables.md")


def stage_plot(p):
    setup_style()
    runs = run_index()
    one, roll, en = load_json("p4_onestep"), load_json("p4_rollout"), load_json("p4_energy")
    misc = load_json("p4_misc") if (RES_DIR / "p4_misc.json").exists() else None
    plot_rollout(roll, runs, p)
    plot_energy(en, runs)
    plot_data_efficiency(one, roll, en, runs)
    if (RES_DIR / "p4_oor_maps.npz").exists():
        plot_oor()
    write_tables(one, roll, en, misc, runs)


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "onestep", "rollout", "energy", "oor", "misc", "plot"])
    ap.add_argument("--runs", default="all", choices=["all", "best"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    p = get_params()
    models = model_list(args.runs)
    print(f"{len(models)} models ({args.runs} runs)", flush=True)
    st = args.stage
    if st in ("all", "onestep"):
        stage_onestep(models, p)
    if st in ("all", "rollout"):
        stage_rollout(models, p)
    if st in ("all", "energy"):
        stage_energy(models, p)
    if st in ("all", "oor"):
        stage_oor(p)
    if st in ("all", "misc"):
        stage_misc(models, p)
    if st in ("all", "plot"):
        stage_plot(p)


if __name__ == "__main__":
    main()
