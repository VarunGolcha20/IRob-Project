# Known issues and methodological risks

This is a running log of things that may be wrong, or that could bias the comparison. Each item gives the evidence (a number from our data), the impact, and what was done about it.

- **Severity:** **High** = could change a headline result. **Medium** = could bias a metric or draw a viva question. **Low** = housekeeping.
- **Status:** **Fixed** = resolved in the code/data already committed. **Decided** = a choice made and implemented. **In plan** = belongs to a later phase; the exact action is written into [PLAN.md](../PLAN.md) in that phase. **Documented** = by design, stated in the docs and the report.

## Summary

| id | issue | severity | status |
|---|---|---|---|
| D1 | The max-based support of R is set by rail-impact rows | High | **Fixed** (density support) + In plan (Phase 4 plots) |
| D2 | Constraint forces as force inputs (total τ_x up to 53 N in R, 168 N in S) | Medium | **Decided:** limit rows dropped from training |
| D3 | The balance region is nearly absent from the data | High | **Fixed:** balance pool S |
| D4 | Training data dominated by near-hanging states | Medium | In plan (Phase 3/4) |
| D5 | Energy-drift "rotating" group outside every training set | Medium | **Fixed:** groups rebuilt and labelled |
| D6 | One episode is a very narrow dataset | Medium | In plan (Phase 3) |
| D7 | Noise-free, exact labels | Low/Medium | In plan (Phase 7) |
| D8 | Row alignment differs from the PDF snippet | Low | Documented |
| D9 | Exploration policies use nominal parameters | Low | Documented |
| D10 | One falling S episode inflates any max-based velocity support | Medium | **Fixed** (same density support as D1) |
| P1 | Initialising DeLaN's damping too high | High | **Fixed:** raw parameter initialised at softplus⁻¹(10⁻⁴) |
| T1 | DeLaN training loss is non-monotone at lr = 10⁻³ | Medium | Mitigated (cosine decay, clipping, best checkpoint); check the curves |
| T2 | λ_inv = 1 chosen on a thin, single-seed margin, at 2× the cost | Medium | **Decided:** keep λ_inv = 1 (rule fixed in advance); report the margin honestly; λ = 0 seeds in Phase 7 |
| T3 | Training is ~4.6× slower than benchmarked (CPU oversubscription) | Medium | Open: benchmark `--threads 1` before relaunching |
| T4 | No mid-run checkpoints: stopping loses in-progress runs | Medium | Partly fixed (resumable per run, detached launch); mid-run checkpointing optional |
| P2 | Energy-drift test asymmetry (DeLaN can set D = 0) | Medium | In plan (Phase 4) |
| A1 | q̈ labels off by 2×10⁻⁴ on rail-limit rows | Low | Moot (limit rows dropped) |
| A2 | No model includes the rail limit | Medium | In plan (Phases 4–5) |

---

## Data generation (Phase 2)

### D1. The "in-range" support of the random data is defined by rail impacts — High
- **Evidence:** 18 of 20 R_train episodes hit the rail; the median first contact is at 4.3 s. The extremes are:

  | boundary | max cos θ | max \|θ̇\| |
  |---|---|---|
  | all rows | −0.229 | 5.29 |
  | 99th percentile | −0.420 | 4.18 |
  | before first rail contact | −0.559 | 4.07 |

  The energy rate is 0.0104 J/s before contact (theory: 0.0107) and 0.0149 J/s overall.
- **Impact:** the max-based box that defines "in range" covers states driven by rare impulsive rail impacts, so it overstates the support.
- **Fixed:**
  - `cartpole.data.in_support` defines a density-based support: a (θ, θ̇) bin of 0.1 rad × 0.25 rad/s must hold at least 5 training rows.
  - Of the 3 677 E_test rows inside the max-based box, only 1 444 are density-supported.
  - None of the OOR-angle or OOR-velocity rows are, so both sets are cleanly out of range.
  - The p99 and pre-contact boundaries are saved in `results/p2_dataset.json` (PART2 §6).
- **In plan (Phase 4):** report error against distance to the density support (a heat map and a binned curve), not only the max-based sets.

### D2. Constraint forces appear as force inputs outside the actuator range — Medium
- **Evidence:** 140 R_train rows (0.7 %) have |τ_x| > 10 N, up to 53 N in total. They are the rail-limit rows, where τ_gen = actuator + constraint. The constraint force alone reaches 61 N in R_train, 67 N in E_train and 178 N in S_train.
- **Decided:** option (a). Both models train on rows without the limit active (`flatten(..., drop_limit=True)`), at a cost of 4 % of R, 0.6 % of E and 2 % of S.
- **Why:**
  - No test-time use ever applies a constraint force: rollouts exclude the rail, and MPPI keeps |x| < 1.6 m.
  - Keeping the rows would favour DeLaN (exactly linear in τ) over the MLP (normalisation stretched by rare 53 N inputs), for a reason unrelated to the physics prior.
- **Test:** after dropping, |τ_x| ≤ 10 N on every row.

### D3. The balance region is almost absent from the training data — High
- **Evidence:** E_train had only 128 rows (0.64 %, from 3 of 20 episodes) with |θ| < 0.2 rad and |θ̇| < 1 rad/s. The pumping law passes upright with speed.
- **Fixed:** added **pool S** (PART2 §4b): an LQR about the upright plus OU noise, 10 train / 3 val / 5 test episodes, starting near upright.
  - The LQR comes from the analytical linearisation, with Bryson's-rule weights. K = [−1.00, −8.60, −1.34, −2.14], and every closed-loop eigenvalue has negative real part.
  - The noise is σ ~ U[0.5, 1.5] per episode, chosen after two checks: σ ∈ [0.2, 0.6] kept |θ| < 0.26, which was too narrow.
  - S_train now has **6 797** balance-region rows.
  - S is used only in regime B (train/val), so regime A's extrapolation test is unchanged.
- **Test:** S_train has more than 5000 balance-region rows.

### D4. The training distribution is dominated by near-hanging states — Medium
- **Evidence:** 61 % of R_train rows have cos θ < −0.95, because every R/E episode starts at rest, hanging.
- **Impact:** an unweighted MSE spends most of its capacity near hanging.
- **In plan:**
  - Phase 4: report the test error per region (hanging, mid, upright, balance, OOR sets) as well as overall.
  - Phase 7 (ablation only): an inverse-density-weighted loss. The main loss stays unweighted, as is standard.

### D5. The energy-drift "rotating" group lay outside every training set — Medium
- **Evidence:** the old "rotating" group (|θ̇₀| ∈ [8.5, 11]) had 29 % of its rows above the E_train max of 8.40 rad/s.
- **Fixed:** the U pool now has 4 groups. "rotating" uses |θ̇₀| ∈ [7.9, 8.3], just above the separatrix (7.81), and the fast states became an explicit "fast_rotating" extrapolation group. Each group carries its density-support coverage per regime (PART2 §5):

  | group | regime A | regime B |
  |---|---|---|
  | small | 100 % | 100 % |
  | large | 15 % | 85 % |
  | rotating | 0 % | 87 % |
  | fast_rotating | 0 % | 1 % |
- **In plan (Phase 4):** drift results are always reported per group with this coverage label.

### D6. A single episode is a very narrow dataset — Medium
- **Evidence:** R episodes vary widely, from no rail contact to contact within 2 s, and from 0.4 to 1.3 rad of maximum pole excursion.
- **In plan (Phase 3):** draw a different random episode subset per seed, use 5 seeds for N = 1 and 3 seeds for N = 5 and 20, and report the per-seed spread.

### D7. Noise-free, exact labels — Low/Medium
- **Evidence:** q̈ is MuJoCo's exact `qacc`, and the states carry no noise.
- **In plan (Phase 7):** a noise ablation with σ_q = 10⁻³ and σ_q̇ = 10⁻², with q̈ obtained by finite differences of the noisy q̇. If there is no time, state it as a limitation.

### D8. Row alignment differs from the PDF snippet — Low, documented
We record the state *before* the step together with the action applied from it, so s_{t+1} = RK4(s_t, u_t) (PART2 §1). The labels are equally exact. The report must say this.

### D9. Exploration policies use nominal parameters — Low, documented
The pumping law (E) and the LQR (S) use the true J and m_p l. They shape the training distribution but are never part of an evaluated model. The report must say this.

### D10. One falling S episode inflates any max-based velocity support — Medium
- **Evidence:** one S_train episode loses the pole and reaches |θ̇| = 9.91, above the E_train max of 8.40. A max-|θ̇| rule then labelled 94 % of the fast-rotating drift group as "in support".
- **Fixed:** all support statements now use the density-based support (D1), which gives 1 % for that group. Max-based numbers stay only as the set definitions.

## Phase 3–5 risks (actions are in the plan)

### P1. Initialising DeLaN's damping too high — High, fixed
softplus(0) = 0.69 would give a damping time constant of 1.6 s, against a true 2200 s, and the data barely constrains D (PART1, P5).
- **Fixed (Phase 3):** `DeLaN.d_raw` is initialised at softplus⁻¹(10⁻⁴). The learned D is reported next to the true values in `results/p3_identification.json`.

### T1. DeLaN training loss is non-monotone at lr = 10⁻³ — Medium, mitigated
- **Evidence:** fitting one batch of 256 rows at a constant lr of 10⁻³, DeLaN's normalised loss goes 14.5 → 7.7×10⁻⁴ (step 1000) → **8.9×10⁻³** (step 2000) → 3.8×10⁻⁴ (step 8000). The MLP decreases steadily. On that narrow batch, which is all near hanging, the learned M also drifts between equally good fits (at step 8000, M₁₂(π) = −0.12 against a true −0.05), a sign of weak identifiability from a narrow θ range.
- **Cause (likely):** q̈ = M⁻¹(…). When M is nearly singular along some direction, small parameter changes produce large changes in q̈, so a fixed step size can overshoot.
- **Mitigation:** cosine decay of the lr to 10⁻⁵, gradient clipping at 10, and keeping the best validation checkpoint (`cartpole/train.py`). The plumbing test now requires a 1000× loss drop instead of a fixed 10⁻³ (`tests/test_models.py`).
- **Action (Phase 3 results):** inspect `figures/p3_training_curves`. Identification is judged only over the training angles, and regime A (θ ≈ π only) is expected to identify M less well than regime B.

### T2. λ_inv = 1 chosen on a thin, single-seed margin — Medium, decided
- **Evidence** (`results/p3_select.json`, DeLaN, seed 0, best validation loss):

  | | λ = 0 | λ = 0.1 | λ = 1 |
  |---|---|---|---|
  | A20 | 1.04×10⁻⁷ | 9.1×10⁻⁸ | 7.9×10⁻⁸ |
  | B20 | 1.94×10⁻⁷ | 2.23×10⁻⁷ | 1.84×10⁻⁷ |
  | mean | 1.49×10⁻⁷ | 1.57×10⁻⁷ | **1.31×10⁻⁷** |

  λ = 1 wins by 12 % on the mean. But λ = 0.1 is *worse* than λ = 0 on B20, so the ordering is not monotone, and seed-to-seed noise is clearly of this size. The inverse loss also doubles DeLaN's cost per step (14.4 against 7.2 ms when benchmarked alone).
- **Decision:** keep λ_inv = 1. The selection rule (lowest mean validation loss over A20 and B20) was fixed in advance in PLAN 3.3, and changing it after seeing the numbers would be a garden-of-forking-paths choice.
- **Report:** state that the EL-residual (inverse) loss gives at most a ~12 % lower validation loss on one seed, which is not shown to be significant, and that it costs 2× compute.
- **In plan (Phase 7):** train λ = 0 at seeds 1–2 for A20 and B20 (4 runs). That gives a real seed-to-seed spread to compare the λ effect against.

### T3. Training is ~4.6× slower than benchmarked — Medium, open
- **Evidence:** a DeLaN (λ = 0) run took 1001 s for 30 000 steps, i.e. 33 ms per step, against 7.2 ms benchmarked alone with 2 threads. The machine has 4 physical cores and 8 hardware threads (i5-11320H). 4 workers × 2 torch threads = 8 threads on 4 cores, so each worker gets about one core and hyper-threading adds little to dense matrix work.
- **Impact:** the full grid takes about 4–5 h of wall time instead of about 1 h.
- **Action (before relaunch):** benchmark 1000 steps with `--workers 4 --threads 1` against `--threads 2`, then relaunch with the faster setting. `scripts/p3_train.py` now has a `--threads` option. The thread count changes only floating-point summation order, not the method, and run names do not depend on it, so finished runs stay valid.

### T4. No mid-run checkpoints — Medium, partly fixed
- **Evidence:**
  - The first launch (17:40) died when VS Code was closed, because it was a child of the editor's shell. Nothing had finished, so about 5 min were lost.
  - The deliberate stop at 18:46 lost 4 in-progress runs, including DeLaN A1 seed 0 after 21.5 min.
- **Fixed:**
  - Long jobs now run detached (`setsid nohup … &`), in their own session, so closing the editor cannot kill them.
  - `scripts/p3_train.py` skips any run whose `weights/<name>.pt` exists, so relaunching resumes at the granularity of whole runs.
  - Every start and finish is logged unbuffered to `results/p3_train.log`.
- **Optional (before relaunch):** save model, optimiser, scheduler and RNG states every 2000 steps, so a resumed run continues bit-for-bit. Worth it only if more stops are expected.

### P2. Fairness of the energy-drift test — Medium
DeLaN can switch its damping off (D = 0); the MLP cannot. The effect is negligible here, because the true time constants are 2200 s and 17 000 s against a 10 s test, so over 10 s the true damping removes at most 0.9 % of the cart's kinetic energy and 0.12 % of the pole's (PART1 P5).
- **In plan (Phase 4):** state this, and also report DeLaN's drift with its learned D kept on.

### A1. Rail-limit q̈ labels deviate by up to 2×10⁻⁴ m/s² — Low, moot
This came from the constraint solver's tolerance (PART1, check 4b). The rows are no longer used for training (D2).

### A2. No model includes the rail limit — Medium
- **In plan:**
  - Phase 4: rollout windows that touch the limit are excluded.
  - Phase 5: MPPI uses a barrier at |x| > 1.6 m. Any episode where the limit becomes active counts as a failure ("cart left the rail").

## Environment and housekeeping

| item | status |
|---|---|
| ROS sourcing leaks into the venv through `PYTHONPATH` | Harmless. Always run with `env -u PYTHONPATH`. |
| MuJoCo 4.0 changed `mj_fullM` to `mj_fullM(m, d, dst)` and removed `d.qM` | Fixed. Teammates **must** use the pinned `requirements.txt`. |
| dm_control 1.0.51 triggers 602 NumPy-2.5 DeprecationWarnings | Harmless. Only that message is filtered in `pytest.ini`. |
| `__pycache__/*.pyc` was committed in the Phase 0/1 commit | Removed from the index; `.gitignore` added. |
| Disk space was low during setup | Torch was installed without pip's cache. Keep weights and videos small. |
| Pumping gains changed from 1.5 to 3 after a test | Documented in PART2 §4. |
| First training launch killed by closing VS Code | Fixed: detached launch with `setsid nohup` (T4). |
