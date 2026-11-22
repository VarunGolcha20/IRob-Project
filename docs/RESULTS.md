# Results log

This is the running record of every result the project produces. Each phase adds a section when its scripts finish.

**Rules for this file:**
- Every number here is copied from a results file or a log. Its source is named under each table.
- Numbers are not rounded more than the table needs.
- Caveats sit next to the numbers they affect and point to [ISSUES.md](ISSUES.md).
- The derivations behind these numbers are in the PART docs. This file only states what was measured.

**Status:**

| phase | results | source |
|---|---|---|
| 1 | Analytical model validated against MuJoCo to round-off | `results/p1_validation.json` |
| 2 | Datasets built; coverage and support measured | `results/p2_dataset.json` |
| 3 | λ_inv selected; 34-run grid finished; parameters identified from the learned M and V | `results/p3_select.json`, `results/p3_training.json`, `results/p3_identification.json` |
| 4 | Open-loop evaluation of all 34 grid models + 3 analytical references | `results/p4_eval.json` |
| 5 | MPPI swing-up: frozen config, 6 models × 20 env seeds | `results/p5_tuning.csv`, `results/p5_mppi_config.json`, `results/p5_closed_loop.json` |

**Notation used in the tables:**
- **A, B:** the training regimes. A uses N random (R) episodes. B uses N R episodes, N energy-pumping (E) episodes and ⌈N/2⌉ LQR-balance (S) episodes.
- **N:** the number of episodes per pool. A1 means regime A with N = 1.
- **s0, s1, …:** training seeds. The seed fixes both which episodes are drawn and the network initialisation.
- **m g l:** m_p g l = 0.4905 J, the energy scale used to normalise energy errors.
- **Validation loss:** the forward loss, i.e. the mean over rows and both axes of ((q̈̂ − q̈)/σ_q̈)², where σ_q̈ is the per-axis standard deviation of the training labels. A value of 10⁻⁶ means an RMS error of 10⁻³ standard deviations per axis.

---

## Phase 1 — Analytical model against MuJoCo

**Check residuals.** The table gives the maximum absolute residual over 6000 rows from random episodes; 369 of those rows have the rail limit active.

| check | quantity compared | max residual |
|---|---|---|
| 1 | M(q) against `mj_fullM` | 2.8×10⁻¹⁷ |
| 2 | Cq̇ + g against `qfrc_bias` | 4.4×10⁻¹⁶ |
| 3 | −Dq̇ against `qfrc_passive` | 0 |
| 4 | q̈ against `qacc`, limit-free rows | 1.4×10⁻¹⁴ |
| 4 | q̈ against `qacc`, limit rows, solver tolerance 10⁻⁸ | 2.1×10⁻⁴ |
| 4b | q̈ against `qacc`, limit rows, solver tolerance 10⁻¹⁵ | 2.1×10⁻¹⁴ |
| 5 | τ̂ against `mj_inverse` | 5.8×10⁻¹⁴ |
| — | energy against `data.energy` | 3.6×10⁻¹⁵ |
| 6 | our RK4 rollout against MuJoCo's RK4 (6 rollouts, up to 6.2 s) | 5.9×10⁻¹⁵ |
| 7 | hand-derived capsule I_c against `body_inertia` | 0 (relative) |

The only residual above round-off, 2.1×10⁻⁴ on limit rows, is the constraint solver's stopping tolerance. Tightening the tolerance to 10⁻¹⁵ removes it (check 4b; ISSUES A1).

**Physical constants measured from the model:**

| quantity | value |
|---|---|
| m_c + m_p, m_p l, J | 1.1 kg, 0.05 kg m, 0.034425 kg m² |
| Thin-rod J (structural-error baseline) | 0.033333 kg m², which is −3.17 % |
| Hanging frequency ω, period | 3.906 rad/s, 1.609 s |
| Upright unstable eigenvalue, error-doubling time | 3.906 /s, 0.177 s |
| Eigenvalues of M at cos θ = ±1, condition number | 0.0321 and 1.102, 34.4 |
| Damping time constants, cart and pole | 2200 s, 17 212 s |

**RK4 energy floor.** The table gives the energy drift of the exact model over 10 s, unforced and undamped, integrated at h = 0.01 s.

| initial state | max \|ΔH\| / (m g l) | max \|Δp_x\| | drift ratio, h against h/2 |
|---|---|---|---|
| swing 0.3 rad | 2.2×10⁻⁹ | 1.3×10⁻¹⁰ | 29.9 |
| swing 0.54 rad | 7.2×10⁻⁹ | 1.2×10⁻⁹ | 27.2 |
| rotating | 1.2×10⁻⁶ | 2.6×10⁻⁷ | 28.2 |

A ratio near 2⁵ = 32 confirms that the drift scales as h⁵. Any learned model's energy drift is measured against this floor.

*Source: `results/p1_validation.json`. Figures: `figures/p1_residuals`, `figures/p1_rk4_vs_mujoco`.*

---

## Phase 2 — Datasets

**Splits.** Each episode lasts 10 s, which is 1000 rows. In the table:
- *Balance rows* have |wrapped θ| < 0.2 rad and |θ̇| < 1 rad/s.
- *Above horizontal* means the fraction of rows with cos θ > 0.

| split | episodes | rows | rows without limit | limit fraction | balance rows | max cos θ | above horizontal | max \|θ̇\| [rad/s] |
|---|---|---|---|---|---|---|---|---|
| R train | 20 | 20 000 | 19 170 | 4.2 % | 0 | −0.229 | 0 | 5.29 |
| R val | 5 | 5000 | 4758 | 4.8 % | 0 | −0.108 | 0 | 5.27 |
| R test | 10 | 10 000 | 9546 | 4.5 % | 0 | −0.457 | 0 | 4.21 |
| E train | 20 | 20 000 | 19 879 | 0.6 % | 128 | 1.000 | 39 % | 8.40 |
| E val | 5 | 5000 | 5000 | 0 | 0 | 0.964 | 29 % | 7.70 |
| E test | 10 | 10 000 | 9968 | 0.3 % | 41 | 1.000 | 36 % | 8.45 |
| S train | 10 | 10 000 | 9786 | 2.1 % | 6797 | 1.000 | 94 % | 9.91 |
| S val | 3 | 3000 | 2840 | 5.3 % | 1739 | 1.000 | 94 % | 9.10 |
| S test | 5 | 5000 | 5000 | 0 | 4241 | 1.000 | 100 % | 2.93 |

**What random forcing reaches:**
- Random forcing never lifts the pole above horizontal. The highest point in R train is 1.34 rad from hanging.
- The cart hits the rail in 18 of the 20 R-train episodes, after a median of 4.29 s.

**Energy gained under random forcing:**

| quantity | value |
|---|---|
| Predicted rate (random-walk theory) | 0.0107 J/s |
| Measured rate, before the first rail contact | 0.0104 J/s (−3 %) |
| Measured rate, whole episode | 0.0149 J/s (rail impacts add energy) |
| Predicted time to reach the upright energy | 92 s, far longer than a 10 s episode |

**Out-of-range (OOR) test rows in E test (10 000 rows):**

| set | definition | rows | rows inside the regime-A density support |
|---|---|---|---|
| OOR-angle | cos θ > 0 | 3596 | 0 |
| OOR-velocity | cos θ < −0.8 and \|θ̇\| > 5.29 rad/s | 1046 | 0 |
| inside the regime-A density support | (θ, θ̇) bin holds ≥ 5 regime-A training rows | 1444 | — |

**Energy-drift initial states (pool U).** There are 10 states per group, unforced and undamped; the rail is never hit.

| group | MuJoCo drift over 10 s, max \|ΔH\|/(m g l) | rows inside support, regime A | rows inside support, regime B |
|---|---|---|---|
| small swing | 5.1×10⁻⁹ | 100 % | 100 % |
| large swing | 8.5×10⁻⁸ | 14.9 % | 85.0 % |
| rotating | 2.2×10⁻⁷ | 0 % | 86.5 % |
| fast rotating | 1.0×10⁻⁶ | 0 % | 1.3 % |

The LQR balance gain is K = [−1.00, −8.60, −1.34, −2.14]. The closed-loop eigenvalues are −7.26 ± 1.36i and −1.36 ± 0.83i, all stable.

*Source: `results/p2_dataset.json`. Figure: `figures/p2_coverage`.*

---

## Phase 3 — Training

Both models have about 34 k parameters: 34 178 for the MLP and 34 310 for DeLaN. Every run uses the same budget of 30 000 Adam steps with a cosine-decayed learning rate. Training stops early after 10 validation checks (5000 steps) without improvement, and the best checkpoint is kept.

**Stage 1: selecting the inverse-loss weight λ_inv.** The table gives DeLaN's best validation loss on seed 0.

| | λ = 0 | λ = 0.1 | λ = 1 |
|---|---|---|---|
| A20 | 1.04×10⁻⁷ | 9.1×10⁻⁸ | 7.9×10⁻⁸ |
| B20 | 1.94×10⁻⁷ | 2.23×10⁻⁷ | 1.84×10⁻⁷ |
| mean | 1.49×10⁻⁷ | 1.57×10⁻⁷ | **1.31×10⁻⁷ (chosen)** |

The margin is 12 % on a single seed, and the ordering is not monotone in λ, so the gain is not shown to be significant (ISSUES T2).

*Source: `results/p3_select.json`.*

**Stage 2: the main grid.** The table gives the best validation loss as the median over seeds, with [min, max] in brackets.

| config | training rows (approx.) | MLP | DeLaN | MLP / DeLaN (medians) |
|---|---|---|---|---|
| A1 (5 seeds) | 1000 | 2.7×10⁻³ [8.2×10⁻⁴, 3.6×10⁻²] | 6.4×10⁻⁵ [3.1×10⁻⁵, 6.0×10⁻⁴] | 43× |
| A5 (3 seeds) | 4800 | 1.1×10⁻⁵ [8.0×10⁻⁶, 1.9×10⁻⁵] | 1.8×10⁻⁷ [4.8×10⁻⁸, 3.8×10⁻⁶] | 60× |
| A20 (3 seeds) | 19 170 | 7.1×10⁻⁶ [6.6×10⁻⁶, 8.9×10⁻⁶] | 4.5×10⁻⁷ [7.9×10⁻⁸, 1.2×10⁻⁶] | 16× |
| B5 (3 seeds) | 12 700 | 6.2×10⁻⁵ [2.8×10⁻⁵, 8.0×10⁻⁵] | 2.3×10⁻⁶ [1.7×10⁻⁷, 9.8×10⁻⁶] | 27× |
| B20 (3 seeds) | 48 835 | 2.5×10⁻⁵ [2.2×10⁻⁵, 2.6×10⁻⁵] | 1.8×10⁻⁷ [1.8×10⁻⁷, 2.9×10⁻⁶] | 140× |

**Per-seed values.** Each entry gives the best validation loss, then (the step of the best checkpoint / the step training stopped at).

| config | model | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| A1 | MLP | 1.5×10⁻³ (4.5k / 9.5k) | 8.2×10⁻⁴ (30k / 30k) | 5.0×10⁻³ (5k / 10k) | 2.7×10⁻³ (28k / 30k) | 3.6×10⁻² (26.5k / 30k) |
| A1 | DeLaN | 6.4×10⁻⁵ (16.5k / 21.5k) | 5.0×10⁻⁵ (15k / 20k) | 8.7×10⁻⁵ (18k / 23k) | 3.1×10⁻⁵ (15.5k / 20.5k) | 6.0×10⁻⁴ (7k / 12k) |
| A5 | MLP | 8.0×10⁻⁶ (30k / 30k) | 1.9×10⁻⁵ (30k / 30k) | 1.1×10⁻⁵ (29.5k / 30k) | | |
| A5 | DeLaN | 4.8×10⁻⁸ (29k / 30k) | 3.8×10⁻⁶ (30k / 30k) | 1.8×10⁻⁷ (29.5k / 30k) | | |
| A20 | MLP | 7.1×10⁻⁶ (30k / 30k) | 8.9×10⁻⁶ (30k / 30k) | 6.6×10⁻⁶ (29.5k / 30k) | | |
| A20 | DeLaN | 7.9×10⁻⁸ (29.5k / 30k) | 1.2×10⁻⁶ (29.5k / 30k) | 4.5×10⁻⁷ (30k / 30k) | | |
| B5 | MLP | 6.2×10⁻⁵ (29k / 30k) | 2.8×10⁻⁵ (30k / 30k) | 8.0×10⁻⁵ (30k / 30k) | | |
| B5 | DeLaN | 1.7×10⁻⁷ (29k / 30k) | **9.8×10⁻⁶ (9.5k / 14.5k)** | **2.3×10⁻⁶ (10k / 15k)** | | |
| B20 | MLP | 2.6×10⁻⁵ (29.5k / 30k) | 2.2×10⁻⁵ (30k / 30k) | 2.5×10⁻⁵ (29.5k / 30k) | | |
| B20 | DeLaN | 1.8×10⁻⁷ (29k / 30k) | **2.9×10⁻⁶ (10k / 15k)** | 1.8×10⁻⁷ (29k / 30k) | | |

**Observations:**
- DeLaN's median validation loss is below the MLP's in all five configurations, by 16× to 140×.
- DeLaN's seeds spread much more widely: up to 79× between seeds (A5), against at most 2.9× for the MLP outside A1.
- The bold DeLaN runs stopped early during a stretch of loss spikes, while the learning rate was still high. They are 10–60× worse than the B seeds that ran the full 30 000 steps. This bias works *against* DeLaN (**ISSUES T3, open**).
- Most N ≥ 5 runs reached their best checkpoint at 29 000–30 000 steps. The step budget, not convergence, therefore sets their final accuracy.
- At the same N, the MLP's validation loss is 3.6× (N = 20) to 5.7× (N = 5) higher in regime B than in regime A. The regime-B validation set holds the harder upright and rotating states, so these numbers are not comparable across regimes.
- These are validation losses only. Test-set metrics come in Phase 4.

*Source: `results/p3_training.json`.*

**Stage 3: parameter identification from the learned structure** (`scripts/p3_structure.py`). Each entry is the relative error |estimate/true − 1|, median over seeds, fitted on the run's own *training angles*; the value in brackets is the same fit over the full circle θ ∈ [−π, π], which for regime A is mostly extrapolation. The last column compares the two independent estimates of m_p l (inertial from M₁₂, gravitational from ∂V/∂θ).

| config | M_tot = m_c + m_p | m_p l (inertial, from M₁₂) | J (from M₂₂) | m_p l (gravitational, from ∂V/∂θ) | inertial / gravitational − 1 |
|---|---|---|---|---|---|
| A1 | 1.7×10⁻³ (7.2×10⁻³) | 2.5×10⁻² (2.0×10⁻¹) | 2.6×10⁻² (5.0×10⁻²) | 2.6×10⁻² (2.0×10⁻²) | 1.1×10⁻³ |
| A5 | 1.4×10⁻⁵ (2.3×10⁻³) | 1.8×10⁻⁴ (2.6×10⁻²) | 1.0×10⁻⁴ (1.1×10⁻²) | 1.9×10⁻⁴ (1.5×10⁻²) | 5.4×10⁻⁵ |
| A20 | 1.4×10⁻⁵ (2.0×10⁻³) | 1.1×10⁻⁴ (4.1×10⁻²) | 4.4×10⁻⁵ (2.3×10⁻²) | 3.4×10⁻⁵ (2.4×10⁻²) | 8.9×10⁻⁵ |
| B5 | 1.7×10⁻⁴ (1.8×10⁻⁴) | 4.2×10⁻³ (3.9×10⁻³) | 3.8×10⁻³ (3.4×10⁻³) | 1.9×10⁻³ (1.9×10⁻³) | 1.4×10⁻³ |
| B20 | 1.4×10⁻⁵ (5.2×10⁻⁵) | 4.8×10⁻⁴ (3.2×10⁻⁴) | 3.6×10⁻⁴ (2.7×10⁻⁴) | 8.5×10⁻⁵ (9.6×10⁻⁵) | 5.5×10⁻⁴ |

| config | learned b_x (true 5×10⁻⁴) | learned b_θ (true 2×10⁻⁶) |
|---|---|---|
| A1 | 6.9×10⁻⁶ | 3.2×10⁻⁵ |
| A5 | 5.9×10⁻⁶ | 1.2×10⁻⁵ |
| A20 | 5.9×10⁻⁶ | 1.0×10⁻⁵ |
| B5 | 2.2×10⁻⁵ | 1.9×10⁻⁵ |
| B20 | 2.0×10⁻⁵ | 7.3×10⁻⁶ |

**Observations:**
- Inside the training support DeLaN recovers M_tot, m_p l and J to 10⁻⁵–10⁻⁴ (N ≥ 5, regime A) and to ≤ 5×10⁻³ (regime B, which spans the whole circle).
- The inertial and gravitational estimates of m_p l agree to 5×10⁻⁵–1.4×10⁻³. This is a physics consistency check the MLP cannot offer.
- Extrapolated to the full circle, regime A drifts to 2–4 % on m_p l and J (figure `p3_learned_structure`: the A20 curve bulges near upright, where it has no data). Regime B stays at its in-support accuracy.
- **Damping is not identified.** The learned b_x is 6×10⁻⁶–2×10⁻⁵, against a true 5×10⁻⁴ (from the initial 10⁻⁴, it moved *away* from the truth). The true damping time constant is 2200 s against 10 s episodes, so the data carry almost no information about it (PART1 P5). This is why the energy-drift test runs DeLaN with D = 0 as well (ISSUES P2).

*Source: `results/p3_identification.json`. Figures: `figures/p3_learned_structure.*`, `figures/p3_training_curves.*`.*

**T3 is still open.** The Phase 4–5 numbers below use the current weights, including the 10 early-stopped runs. Phase 4 shows that T3 does not change any ordering (ISSUES T3, new evidence).

---

## Phase 4 — Open-loop evaluation

All models run in float64 through the same RK4 (h = 0.01 s) in the torch-free backend `cartpole/npdyn.py`, which reproduces each checkpoint's stored validation loss to ≤ 1.7×10⁻⁴ relative (float32 vs float64; `tests/test_npdyn.py`). Ground truth is the recorded MuJoCo data. Definitions are in [PART4_EVAL.md](PART4_EVAL.md). The normaliser for nRMSE is the per-axis std of q̈ over all test rows (4.80 m/s², 9.66 rad/s²; `normaliser_qdd_std`).

**One-step nRMSE** (RMS over both axes, median over seeds):

| model | all | hanging | mid | upright | balance | OOR-angle | OOR-velocity | in support | out of support |
|---|---|---|---|---|---|---|---|---|---|
| MLP A1 | 4.6×10⁻¹ | 4.2×10⁻² | 3.4×10⁻¹ | 8.5×10⁻¹ | 8.8×10⁻¹ | 5.4×10⁻¹ | 1.3×10⁻¹ | 6.6×10⁻³ | 5.0×10⁻¹ |
| DeLaN A1 | 1.3×10⁻¹ | 2.4×10⁻³ | 6.5×10⁻² | 2.4×10⁻¹ | 2.6×10⁻¹ | 1.1×10⁻¹ | 6.7×10⁻³ | 7.5×10⁻⁴ | 1.4×10⁻¹ |
| MLP A5 | 1.5×10⁻¹ | 1.2×10⁻² | 7.9×10⁻² | 2.8×10⁻¹ | 3.0×10⁻¹ | 1.3×10⁻¹ | 3.7×10⁻² | 9.9×10⁻⁴ | 1.8×10⁻¹ |
| DeLaN A5 | 2.6×10⁻² | 2.5×10⁻⁴ | 1.5×10⁻² | 5.1×10⁻² | 5.4×10⁻² | 2.5×10⁻² | 6.8×10⁻⁴ | 1.6×10⁻⁴ | 3.3×10⁻² |
| MLP A20 | 1.4×10⁻¹ | 1.2×10⁻² | 8.0×10⁻² | 2.7×10⁻¹ | 2.9×10⁻¹ | 1.4×10⁻¹ | 3.9×10⁻² | 1.3×10⁻³ | 1.9×10⁻¹ |
| DeLaN A20 | 4.1×10⁻² | 5.2×10⁻⁴ | 2.4×10⁻² | 7.7×10⁻² | 8.3×10⁻² | 3.9×10⁻² | 1.5×10⁻³ | 2.8×10⁻⁴ | 5.4×10⁻² |
| MLP B5 | 3.6×10⁻³ | 2.4×10⁻³ | 5.3×10⁻³ | 2.6×10⁻³ | 2.2×10⁻³ | 4.8×10⁻³ | 5.9×10⁻³ | 1.9×10⁻³ | 5.1×10⁻³ |
| DeLaN B5 | 7.6×10⁻⁴ | 3.9×10⁻⁴ | 1.1×10⁻³ | 8.0×10⁻⁴ | 5.8×10⁻⁴ | 1.6×10⁻³ | 7.8×10⁻⁴ | 4.5×10⁻⁴ | 1.0×10⁻³ |
| MLP B20 | 2.3×10⁻³ | 1.8×10⁻³ | 3.3×10⁻³ | 1.8×10⁻³ | 1.4×10⁻³ | 3.0×10⁻³ | 4.2×10⁻³ | 2.1×10⁻³ | 3.9×10⁻³ |
| DeLaN B20 | 3.4×10⁻⁴ | 2.6×10⁻⁴ | 4.5×10⁻⁴ | 3.2×10⁻⁴ | 2.4×10⁻⁴ | 5.3×10⁻⁴ | 6.6×10⁻⁴ | 2.9×10⁻⁴ | 6.2×10⁻⁴ |
| thin_rod | 2.4×10⁻² | 2.2×10⁻² | 3.1×10⁻² | 1.5×10⁻² | 1.6×10⁻² | 2.7×10⁻² | 1.9×10⁻² | — | — |
| plus1pct | 9.6×10⁻³ | 1.1×10⁻² | 7.9×10⁻³ | 9.0×10⁻³ | 9.7×10⁻³ | 6.3×10⁻³ | 8.1×10⁻³ | — | — |

Rows per region: all 24514, hanging 11179, mid 7313, upright 6022, R_test 9546, balance (S_test) 5000, OOR-angle (E_test) 3596, OOR-velocity (E_test) 1046.

**Rollout RMSE of wrapped θ [rad]** (median over windows, then median over seeds) and **valid-prediction time [s]** (mean over windows, capped at 2 s):

| model | R 0.5 s | R 1 s | R 2 s | E 0.5 s | E 1 s | E 2 s | S 0.5 s | S 1 s | S 2 s | VPT R | VPT E | VPT S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytical (exact) | 0 | 1.0×10⁻¹⁶ | 3.2×10⁻¹⁶ | 0 | 2.1×10⁻¹⁶ | 6.3×10⁻¹⁶ | 2.3×10⁻¹⁷ | 1.3×10⁻¹⁶ | 4.5×10⁻¹⁵ | 2.00 | 2.00 | 2.00 |
| analytical, thin rod | 5.2×10⁻³ | 1.1×10⁻² | 2.1×10⁻² | 1.8×10⁻² | 4.2×10⁻² | 1.0×10⁻¹ | 4.1×10⁻³ | 2.1×10⁻² | 7.5×10⁻¹ | 1.98 | 1.23 | 1.14 |
| analytical, +1 % params | 9.2×10⁻⁴ | 1.7×10⁻³ | 3.1×10⁻³ | 2.7×10⁻³ | 6.9×10⁻³ | 1.5×10⁻² | 2.1×10⁻³ | 1.2×10⁻² | 4.1×10⁻¹ | 2.00 | 1.92 | 1.34 |
| MLP A1 | 1.2×10⁻³ | 2.7×10⁻³ | 3.4×10⁻³ | 4.7×10⁻² | 1.5×10⁻¹ | 6.0×10⁻¹ | 2.7×10⁻¹ | 1.4×10⁰ | 1.5×10⁰ | 1.99 | 0.70 | 0.25 |
| DeLaN A1 | 1.1×10⁻⁴ | 2.7×10⁻⁴ | 5.2×10⁻⁴ | 8.0×10⁻³ | 3.1×10⁻² | 1.0×10⁻¹ | 8.6×10⁻² | 4.3×10⁻¹ | 1.3×10⁰ | 2.00 | 1.31 | 0.45 |
| MLP A5 | 8.7×10⁻⁵ | 1.8×10⁻⁴ | 2.6×10⁻⁴ | 6.4×10⁻³ | 2.5×10⁻² | 7.2×10⁻² | 1.0×10⁻¹ | 5.1×10⁻¹ | 1.3×10⁰ | 2.00 | 1.36 | 0.41 |
| DeLaN A5 | 3.4×10⁻⁵ | 7.2×10⁻⁵ | 1.3×10⁻⁴ | 1.9×10⁻³ | 7.7×10⁻³ | 2.5×10⁻² | 2.0×10⁻² | 1.0×10⁻¹ | 1.2×10⁰ | 2.00 | 1.70 | 0.78 |
| MLP A20 | 8.4×10⁻⁵ | 1.6×10⁻⁴ | 2.6×10⁻⁴ | 4.6×10⁻³ | 2.0×10⁻² | 6.6×10⁻² | 1.1×10⁻¹ | 7.3×10⁻¹ | 1.3×10⁰ | 2.00 | 1.39 | 0.37 |
| DeLaN A20 | 5.2×10⁻⁵ | 1.1×10⁻⁴ | 2.0×10⁻⁴ | 3.2×10⁻³ | 1.2×10⁻² | 4.0×10⁻² | 2.9×10⁻² | 1.6×10⁻¹ | 1.2×10⁰ | 2.00 | 1.57 | 0.69 |
| MLP B5 | 1.9×10⁻⁴ | 3.8×10⁻⁴ | 5.7×10⁻⁴ | 6.6×10⁻⁴ | 1.8×10⁻³ | 3.9×10⁻³ | 4.2×10⁻⁴ | 2.7×10⁻³ | 1.1×10⁻¹ | 2.00 | 1.96 | 1.63 |
| DeLaN B5 | 5.4×10⁻⁵ | 1.1×10⁻⁴ | 2.1×10⁻⁴ | 2.4×10⁻⁴ | 6.7×10⁻⁴ | 1.4×10⁻³ | 2.6×10⁻⁴ | 1.6×10⁻³ | 5.8×10⁻² | 2.00 | 2.00 | 1.80 |
| MLP B20 | 1.3×10⁻⁴ | 2.6×10⁻⁴ | 4.2×10⁻⁴ | 4.7×10⁻⁴ | 1.1×10⁻³ | 2.9×10⁻³ | 3.1×10⁻⁴ | 1.9×10⁻³ | 7.2×10⁻² | 2.00 | 1.99 | 1.73 |
| DeLaN B20 | 3.6×10⁻⁵ | 7.3×10⁻⁵ | 7.7×10⁻⁵ | 7.2×10⁻⁵ | 1.7×10⁻⁴ | 3.8×10⁻⁴ | 9.9×10⁻⁵ | 6.7×10⁻⁴ | 2.5×10⁻² | 2.00 | 2.00 | 1.93 |

Windows: R_test 214, E_test 321, S_test 165.

**Energy drift**, max over 10 s of |H_true(ŝ) − H_true(s₀)| / (m_p g l), worst state of each group, median over seeds; DeLaN with D = 0 (learned D in brackets):

| model | small | large | rotating | fast_rotating |
|---|---|---|---|---|
| analytical (exact) | 5.1×10⁻⁹ | 8.5×10⁻⁸ | 2.2×10⁻⁷ | 1.0×10⁻⁶ |
| analytical, thin rod | 3.1×10⁻³ | 6.8×10⁻² | 7.0×10⁻² | 7.0×10⁻² |
| analytical, +1 % params | 3.7×10⁻⁴ | 8.0×10⁻³ | 8.3×10⁻³ | 8.4×10⁻³ |
| MLP A1 | 1.3×10⁻² | 1.1×10² | 1.6×10² | 2.5×10² |
| DeLaN A1 | 1.5×10⁻⁴ (1.2×10⁻³) | 4.0×10⁻² (4.6×10⁻²) | 5.9×10⁻² (8.1×10⁻²) | 1.8×10⁻¹ (2.4×10⁻¹) |
| MLP A5 | 6.0×10⁻⁴ | 5.5×10⁰ | 9.6×10⁰ | 1.4×10¹ |
| DeLaN A5 | 2.8×10⁻⁵ (4.2×10⁻⁴) | 3.2×10⁻² (3.2×10⁻²) | 4.4×10⁻² (5.0×10⁻²) | 1.1×10⁻¹ (1.3×10⁻¹) |
| MLP A20 | 1.2×10⁻³ | 8.5×10⁰ | 9.2×10⁰ | 1.0×10¹ |
| DeLaN A20 | 2.6×10⁻⁵ (3.6×10⁻⁴) | 4.8×10⁻² (4.8×10⁻²) | 6.8×10⁻² (7.3×10⁻²) | 1.9×10⁻¹ (2.1×10⁻¹) |
| MLP B5 | 1.5×10⁻³ | 1.4×10⁻² | 7.6×10⁻² | 2.5×10⁰ |
| DeLaN B5 | 3.0×10⁻⁵ (7.0×10⁻⁴) | 6.3×10⁻⁴ (7.4×10⁻³) | 2.6×10⁻³ (1.3×10⁻²) | 1.4×10⁻² (4.4×10⁻²) |
| MLP B20 | 1.9×10⁻³ | 1.7×10⁻² | 2.2×10⁻² | 2.2×10⁰ |
| DeLaN B20 | 4.8×10⁻⁵ (2.6×10⁻⁴) | 4.5×10⁻⁴ (2.9×10⁻³) | 6.3×10⁻⁴ (4.3×10⁻³) | 2.2×10⁻³ (1.3×10⁻²) |

DeLaN's drift of its *own* energy H (D = 0), fast_rotating group, median over seeds: A1 9.0×10⁻⁷, A5 9.9×10⁻⁷, A20 9.0×10⁻⁷, B5 1.0×10⁻⁶, B20 1.0×10⁻⁶.

**Inverse dynamics** (DeLaN, all test rows, median over seeds):

| config | τ_x RMSE [N] | τ_x nRMSE | τ_θ RMS [N m] (should be 0) |
|---|---|---|---|
| A1 | 1.2×10⁻¹ | 2.5×10⁻² | 6.2×10⁻² |
| A5 | 2.6×10⁻² | 5.3×10⁻³ | 1.1×10⁻² |
| A20 | 2.1×10⁻² | 4.2×10⁻³ | 1.8×10⁻² |
| B5 | 2.0×10⁻³ | 3.9×10⁻⁴ | 3.5×10⁻⁴ |
| B20 | 5.9×10⁻⁴ | 1.2×10⁻⁴ | 1.5×10⁻⁴ |

**Compute** (numpy float64, batch 512, one q̈ call): analytical 26 µs, MLP 2479 µs, DeLaN 4788 µs.

**Observations** (checked against the expectations written in PLAN Phase 4 before running):
- **One-step:** DeLaN's median is below the MLP's in every region and configuration, by 3–54×. In regime A20 the gap is 3.5× upright and 25× on OOR-velocity. *Expected:* yes for OOR-velocity; OOR-angle was "open" and DeLaN wins it 3.6× (A20).
- **Worst DeLaN seed vs best MLP seed:** DeLaN wins in 36 of 40 (configuration × metric) cells (one-step all / upright / OOR-angle / OOR-velocity, rollout E 1 s and S 1 s, drift large and fast_rotating). The 4 exceptions all involve seeds with T3 early stopping or the A5 seed-1 outlier.
- **Rollouts:** DeLaN B20 beats the analytical model with every parameter 1 % off, at every horizon and pool (e.g. S_test 2 s: 2.5×10⁻² vs 4.1×10⁻¹ rad). Near upright every learned model degrades sharply between 1 s and 2 s, as the λ = 3.9 /s amplification predicts. *Expected:* yes.
- **Valid-prediction time** saturates at the 2 s cap on R_test (hanging, stable), so it only discriminates on E_test and S_test.
- **Energy drift:** DeLaN's drift (median over seeds) is 21–2800× smaller than the MLP's in every group and configuration. The MLP's regime-A models blow up on the large/rotating groups (drift 5–250 m_p g l), while DeLaN A20 stays at 0.05–0.2. DeLaN's drift of its own H stays at the RK4 floor (~10⁻⁶), so its remaining true-energy drift comes from the mismatch between learned and true H, not from accumulation. *Expected:* yes.
- **Learned damping hurts the drift test** (bracketed values, up to 23×), because the learned D is wrong (Phase 3 stage 3). D = 0 is the fair setting (ISSUES P2).
- **Data efficiency:** on energy drift DeLaN's margin is by far the largest at N = 1 (2700× on "large", against 170–180× at N = 5 and 20). On one-step error it is not: 3.7× at N = 1, 5.8× at N = 5, 3.4× at N = 20. From N = 5 to N = 20 neither model improves much, so the 30 000-step budget, not the data, limits them (as stage 2 already suggested). *Expected:* partly. It holds for drift, not for one-step error.
- **Inverse dynamics:** DeLaN B20 predicts τ_x to 5.9×10⁻⁴ N RMS and leaves a θ-row residual of 1.5×10⁻⁴ N m (should be 0).
- **Compute:** in numpy a DeLaN call costs 1.9× an MLP call. But DeLaN's M, ∂M/∂θ and ∂V/∂θ depend on θ alone, so DeLaN can be tabulated once and then costs ~4× the analytical model (Phase 5).

*Source: `results/p4_eval.json` (tables generated by `scripts/results_tables.py p4`). Figures: `figures/p4_data_efficiency.*`, `p4_rollout.*`, `p4_energy_drift.*`, `p4_oor_heatmap.*`, `p4_error_vs_support_distance.*`.*

## Phase 5 — MPPI swing-up

Every model drives the **same** frozen MPPI configuration (PLAN 5.1 defaults: K = 512, H = 75, Δt_plan = 0.02 s, σ = 0.5, AR(1) 0.8, ESS ∈ [0.1K, 0.2K], no LQR), replanning at 50 Hz. Definitions are in [PART5_MPPI.md](PART5_MPPI.md).
- **Tuning:** the plan-default configuration passed 10/10 on the analytical model (seeds 100–109) at the first attempt, so no other configuration was tried (`results/p5_tuning.csv`).
- **Evaluation:** env seeds 0–19, whose initial states are identical to dm_control's swing-up resets. For each learned model, the training seed with the best validation loss is used.
- **Plant:** the analytical model + RK4, equal to MuJoCo off the rail to 10⁻¹⁴. The rail is never reached (0 contacts), so the missing rail model plays no role (ISSUES E2).

| model | weights | success | Wilson 95 % | held to end | Wilson 95 % (held) | time balanced after first balance (median) | rail contacts | time-to-balance median [IQR] (s) | max \|x\| (m) | dm_control return (median) | ms per plan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| analytical | — | 20/20 | [0.84, 1.00] | 20/20 | [0.84, 1.00] | 1.000 | 0 | 1.97 [1.94, 1.99] | 0.94 | 848 | 13.1 |
| thin_rod | — | 20/20 | [0.84, 1.00] | 20/20 | [0.84, 1.00] | 1.000 | 0 | 1.93 [1.89, 1.96] | 0.93 | 852 | 11.7 |
| DeLaN-B20 | delan_B20_s2_inv1 | 20/20 | [0.84, 1.00] | 20/20 | [0.84, 1.00] | 1.000 | 0 | 1.97 [1.94, 1.99] | 0.94 | 848 | 32.0 |
| MLP-B20 | mlp_B20_s1 | 20/20 | [0.84, 1.00] | 20/20 | [0.84, 1.00] | 1.000 | 0 | 1.99 [1.95, 2.00] | 0.93 | 847 | 285.1 |
| DeLaN-A20 | delan_A20_s0_inv1 | 20/20 | [0.84, 1.00] | 20/20 | [0.84, 1.00] | 1.000 | 0 | 1.99 [1.95, 2.00] | 0.93 | 848 | 31.1 |
| MLP-A20 | mlp_A20_s2 | 20/20 | [0.84, 1.00] | 8/20 | [0.22, 0.61] | 0.674 | 0 | 2.19 [2.10, 3.38] | 1.54 | 638 | 280.4 |

"Held to end" = success **and** balanced over the entire final 2 s. "Time balanced after first balance" = the fraction of steps after the first balanced instant at which the pole is balanced (ISSUES E5).

**Observations:**
- **Every model passes the plan's success criterion 20/20.** That includes MLP-A20 and DeLaN-A20, which never saw a state above horizontal. With n = 20 the Wilson intervals all equal [0.84, 1.00], so this metric separates nothing.
- **The stricter criterion separates the regime-A models.** DeLaN-A20 keeps the pole up to the end of all 20 episodes. MLP-A20 does so in only 8/20 (Wilson [0.22, 0.61], disjoint from DeLaN-A20's [0.84, 1.00]) and is balanced only 67 % of the time after first reaching upright: it balances, falls, swings up again and falls again (figure `p5_time_to_balance`, right). It also swings the cart out to 1.54 m (vs 0.93 m) and loses 25 % of the dm_control return (638 vs 848).
- **DeLaN-A20 is indistinguishable from the analytical model in closed loop** (time-to-balance 1.99 vs 1.97 s, return 848 vs 848), even though its open-loop upright error is 7.7×10⁻² nRMSE. Its learned structure extrapolates well enough for 1.5 s plans with 50 Hz feedback. The MLP-A20 (2.7×10⁻¹ upright, and energy drift up to 10 m_p g l) does not.
- **With regime-B data both learned models match the analytical model.** Upright data closes the gap for the MLP.
- **Compute:** DeLaN in the planner costs 32 ms per plan (θ lookup table), against 285 ms for the MLP (9×) and 13 ms for the analytical model. At 50 Hz the real-time budget is 20 ms. Only the analytical model fits it in this single-threaded numpy implementation; DeLaN would need 1.6× more speed and the MLP 14×.

*Source: `results/p5_closed_loop.json`, per-episode `results/p5_runs/*.json` (tables generated by `scripts/results_tables.py p5`). Figures: `figures/p5_success.*`, `p5_time_to_balance.*`, `p5_trajectories.*`.*
