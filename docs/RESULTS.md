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
| 3 | λ_inv selected; 34-run training grid finished. Identification is pending (`scripts/p3_structure.py` not yet run) | `results/p3_select.json`, `results/p3_training.json` |
| 4 | — | — |
| 5 | — | — |

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

**Pending for Phase 3:**
- Parameter identification and the learned-structure figure (`scripts/p3_structure.py`).
- The training-curve figure.
- The decision on T3.

---

## Phase 4 — Open-loop evaluation

*Not started.*

## Phase 5 — MPPI swing-up

*Not started.*
