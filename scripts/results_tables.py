"""Print the markdown tables of docs/RESULTS.md (Phases 3-5) straight from the results files,
so no number is transcribed by hand.

Usage:  python scripts/results_tables.py [p3 | p4 | p5 | all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
CFGS = ["A1", "A5", "A20", "B5", "B20"]


def e(v, d=1):
    if v is None:
        return "—"
    if v == 0:
        return "0"
    m, x = f"{v:.{d}e}".split("e")
    x = int(x)
    sup = str(x).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}×10{sup}"


def p3():
    d = json.loads((RES / "p3_identification.json").read_text())
    print("| config | M_tot = m_c + m_p | m_p l (inertial, from M₁₂) | J (from M₂₂) | m_p l (gravitational, from ∂V/∂θ) | inertial / gravitational − 1 |")
    print("|---|---|---|---|---|---|")
    for c in CFGS:
        rs = [v for v in d["runs"].values() if f"{v['regime']}{v['N']}" == c]

        def cell(k):
            a = np.median([abs(r["rel_error"][k]) for r in rs])
            b = np.median([abs(r["full_circle"][k]) for r in rs])
            return f"{e(a)} ({e(b)})"
        ig = np.median([abs(r["inertial_vs_gravitational"]) for r in rs])
        print(f"| {c} | {cell('M_tot')} | {cell('mpl_inertial')} | {cell('J')} | {cell('mpl_gravitational')} | {e(ig)} |")
    print()
    print("| config | learned b_x (true 5×10⁻⁴) | learned b_θ (true 2×10⁻⁶) |")
    print("|---|---|---|")
    for c in CFGS:
        rs = [v for v in d["runs"].values() if f"{v['regime']}{v['N']}" == c]
        print(f"| {c} | {e(np.median([r['estimate']['b_x'] for r in rs]))} | {e(np.median([r['estimate']['b_th'] for r in rs]))} |")


def p4():
    r = json.loads((RES / "p4_eval.json").read_text())
    S, M = r["summary"], r["models"]

    def med(k, key):
        v = S[k][key]
        return v["median"] if isinstance(v, dict) else v

    regs = [("all", "all"), ("hanging", "hanging"), ("mid", "mid"), ("upright", "upright"),
            ("balance (S_test)", "balance"), ("OOR-angle (E_test)", "OOR-angle"),
            ("OOR-velocity (E_test)", "OOR-velocity"), ("in support", "in support"), ("out of support", "out of support")]
    print("**One-step nRMSE** (RMS over both axes, median over seeds):\n")
    print("| model | " + " | ".join(b for _, b in regs) + " |")
    print("|---|" + "---|" * len(regs))
    for c in CFGS:
        for kind in ("mlp", "delan"):
            k = f"{kind}_{c}"
            print(f"| {'MLP' if kind == 'mlp' else 'DeLaN'} {c} | " +
                  " | ".join(e(med(k, f'one_step_nrmse::{a}')) if f'one_step_nrmse::{a}' in S[k] else "—" for a, _ in regs) + " |")
    for ref in ("thin_rod", "plus1pct"):
        o = M[ref]["one_step"]
        print(f"| {ref} | " + " | ".join(e(np.hypot(o[a]['nrmse_xdd'], o[a]['nrmse_thdd']) / np.sqrt(2)) if a in o else "—"
                                          for a, _ in regs) + " |")
    print(f"\nRows per region: " + ", ".join(f"{k} {v}" for k, v in r["regions"].items()) + ".\n")

    print("**Rollout RMSE of wrapped θ [rad]** (median over windows, then median over seeds) and **valid-prediction time [s]** (mean over windows, capped at 2 s):\n")
    print("| model | R 0.5 s | R 1 s | R 2 s | E 0.5 s | E 1 s | E 2 s | S 0.5 s | S 1 s | S 2 s | VPT R | VPT E | VPT S |")
    print("|---|" + "---|" * 12)
    keys = ["analytical", "thin_rod", "plus1pct"] + [f"{k}_{c}" for c in CFGS for k in ("mlp", "delan")]
    names = {"analytical": "analytical (exact)", "thin_rod": "analytical, thin rod", "plus1pct": "analytical, +1 % params"}
    for k in keys:
        nm = names.get(k, k.replace("mlp_", "MLP ").replace("delan_", "DeLaN "))
        print(f"| {nm} | " + " | ".join(e(med(k, f'rollout_theta::{p}::{h}')) for p in ("R_test", "E_test", "S_test")
                                        for h in ("0.5s", "1s", "2s")) +
              " | " + " | ".join(f"{med(k, f'vpt::{p}'):.2f}" for p in ("R_test", "E_test", "S_test")) + " |")
    print(f"\nWindows: " + ", ".join(f"{k} {v}" for k, v in r["n_windows"].items()) + ".\n")

    print("**Energy drift**, max over 10 s of |H_true(ŝ) − H_true(s₀)| / (m_p g l), worst state of each group, median over seeds; DeLaN with D = 0 (learned D in brackets):\n")
    groups = ("small", "large", "rotating", "fast_rotating")
    print("| model | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * 4)
    for k in keys:
        nm = names.get(k, k.replace("mlp_", "MLP ").replace("delan_", "DeLaN "))
        cells = []
        for g in groups:
            c = e(med(k, f"drift::{g}"))
            if k.startswith("delan"):
                rs = [v for v in M.values() if v.get("kind") == "delan" and f"delan_{v['regime']}{v['N']}" == k]
                c += f" ({e(np.median([v['drift_learned_D'][g]['dH_true_max'] for v in rs]))})"
            cells.append(c)
        print(f"| {nm} | " + " | ".join(cells) + " |")
    own = {c: np.median([v["drift"]["fast_rotating"]["dH_own_max"] for v in M.values()
                         if v.get("kind") == "delan" and f"{v['regime']}{v['N']}" == c]) for c in CFGS}
    print("\nDeLaN's drift of its *own* energy H (D = 0), fast_rotating group, median over seeds: " +
          ", ".join(f"{c} {e(v)}" for c, v in own.items()) + ".\n")

    print("**Inverse dynamics** (DeLaN, all test rows, median over seeds):\n")
    print("| config | τ_x RMSE [N] | τ_x nRMSE | τ_θ RMS [N m] (should be 0) |")
    print("|---|---|---|---|")
    for c in CFGS:
        rs = [v for v in M.values() if v.get("kind") == "delan" and f"{v['regime']}{v['N']}" == c]
        print(f"| {c} | {e(np.median([v['inverse']['tau_x_rmse_N'] for v in rs]))} | "
              f"{e(np.median([v['inverse']['tau_x_nrmse'] for v in rs]))} | "
              f"{e(np.median([v['inverse']['tau_theta_rms_Nm'] for v in rs]))} |")
    t = r["compute_us_per_qdd_batch512"]
    print(f"\n**Compute** (numpy float64, batch 512, one q̈ call): analytical {t['analytical']:.0f} µs, "
          f"MLP {t['mlp']:.0f} µs, DeLaN {t['delan']:.0f} µs.")


def p5():
    r = json.loads((RES / "p5_closed_loop.json").read_text())
    print("| model | weights | success | Wilson 95 % | held to end | Wilson 95 % (held) | time balanced after first balance (median) | rail contacts | time-to-balance median [IQR] (s) | max \\|x\\| (m) | dm_control return (median) | ms per plan |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, v in r["models"].items():
        iqr = v["time_to_balance_iqr"]
        ttb = f"{v['time_to_balance_median']:.2f} [{iqr[0]:.2f}, {iqr[1]:.2f}]" if iqr else "—"
        print(f"| {m} | {v['weights'] or '—'} | {v['n_success']}/{v['n']} | [{v['wilson95'][0]:.2f}, {v['wilson95'][1]:.2f}] | "
              f"{v['n_held_to_end']}/{v['n']} | [{v['wilson95_held'][0]:.2f}, {v['wilson95_held'][1]:.2f}] | "
              f"{v['frac_time_balanced_after_first_median']:.3f} | {v['n_rail']} | {ttb} | {v['max_abs_x_max']:.2f} | {v['dmc_return_median']:.0f} | {v['ms_per_plan_mean']:.1f} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("p3", p3), ("p4", p4), ("p5", p5)):
        if which in (name, "all"):
            print(f"\n<!-- {name} -->\n")
            f()
