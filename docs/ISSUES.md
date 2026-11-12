# Known issues and methodological risks

This is a running log of things that may be wrong, or that could bias the comparison. Every item gives the evidence (a number from our data), the impact, and the proposed action. "Open" items must be resolved before the phase named in the action.

Severity: **High** = could change a headline result. **Medium** = could bias a metric or draw a viva question. **Low** = housekeeping.

---

## Data generation (Phase 2)

### D1. The "in-range" support of the random data is defined by rail impacts — High, open
- **Evidence:** 18 of 20 R_train episodes hit the rail; the median first contact is at 4.3 s. Before the first contact, max |θ̇| = 4.07 rad/s and max cos θ = −0.559. Over whole episodes they are 5.29 rad/s and −0.229. The energy growth rate is 0.0104 J/s before contact (theory: 0.0107) and 0.0149 J/s overall (`results/p2_dataset.json`).
- **Impact:** the R pool is not "random forcing" alone. Its widest states come from impulsive constraint forces of up to 53 N, five times the actuator limit. The OOR boundaries (cos θ > −0.229, |θ̇| > 5.29) are therefore set by these rare impact-driven rows, which makes the "in-range" region look larger than the bulk of the data supports.
- **Action (Phase 4):** also report the boundaries at the 99th percentile (cos θ = −0.420, |θ̇| = 4.18). Plot error against distance from the training support (the planned heat map) rather than relying only on a max-based cut. State clearly in the report that R includes rail impacts.

### D2. Constraint forces appear as force inputs outside the actuator range — Medium, open
- **Evidence:** 140 R_train rows (0.7 %) have |τ_x| > 10 N, up to 53 N. They are the rail-limit rows, where τ_gen = actuator + constraint.
- **Impact:**
  - DeLaN is exactly linear in τ, so these rows are harmless and informative for it.
  - The MLP sees τ values it will never see in control, and they stretch its input normalisation (std of τ).
  - The limit rows are also impulsive and only 4 % of the data.
- **Action (Phase 3):** decide between (a) training both models on limit-free rows only, or (b) keeping τ_gen with normalisation statistics from limit-free rows. **Recommendation: (a).** It is simplest and fair, it costs 4 % of R and 0.6 % of E, and both test-time uses (rollouts and MPPI) exclude the rail anyway. Whatever we choose must apply to both models.

### D3. The balance region is almost absent from the training data — High, open
- **Evidence:** in E_train, states with |θ| < 0.2 rad and |θ̇| < 1 rad/s number only **128 rows (0.64 %), from 3 of 20 episodes**. The pumping law takes the pole through upright with speed, and only near-separatrix targets (E_ref ≈ m_p g l) approach the upright slowly.
- **Impact:** MPPI balancing happens exactly in this region. Even regime-B models may be poor there, and the closed-loop comparison would then measure extrapolation into the balance region rather than model quality.
- **Action (before Phase 5):** add a small **balance pool** to E: start near upright (|θ| < 0.3, small velocities), stabilise with an LQR built from the analytical model plus OU noise, and add 5 train, 2 val and 3 test episodes. This is the same "controller rollouts" excitation the instructions list, and the same policy-uses-nominal-model reasoning as the pumping law applies. Keep it out of regime A, so the extrapolation story still holds.

### D4. The training distribution is dominated by near-hanging states — Medium, open
- **Evidence:** 61 % of R_train rows have cos θ < −0.95. Every episode starts at rest, hanging (figure p2_coverage, panel c).
- **Impact:** an unweighted MSE spends most of its capacity near hanging. Errors elsewhere are under-penalised, which hurts rollouts that leave the bottom.
- **Action (Phase 3):** report the test error per region, not just the overall average. Consider an inverse-density weight as an ablation only. The main loss stays unweighted, so the comparison is standard.

### D5. The energy-drift "rotating" group lies outside every training set — Medium, open
- **Evidence:** the U "rotating" group starts at |θ̇₀| ∈ [8.5, 11] rad/s. 29 % of its rows exceed the E_train maximum of 8.40 rad/s. The "large" group spends 22 % of its time above horizontal, which is out of range for regime A.
- **Impact:** the drift results for the "rotating" group (and for "large" under regime A) measure extrapolation, not conservation. That is fine if labelled, misleading if not.
- **Action (Phase 4):** label each drift group as in-support or out-of-support per regime. Optionally regenerate "rotating" with |θ̇₀| ∈ [7.9, 8.3], just above the separatrix (7.81) and inside E support, and keep the fast set as an explicit extrapolation test.

### D6. A single episode is a very narrow dataset — Medium, open
- **Evidence:** episodes differ widely. In R, rail contact ranges from never to within 2 s, and the max pole excursion ranges from 0.4 to 1.3 rad.
- **Impact:** the N = 1 point of the data-efficiency curve depends heavily on which episode is drawn. Three seeds give only a crude spread.
- **Action (Phase 3):** draw a different random episode subset per seed (already planned), use 5 seeds for N = 1 because training is cheap, and report the per-seed spread.

### D7. Noise-free, exact labels — Low/Medium (discussion item)
- **Evidence:** q̈ is MuJoCo's exact `qacc`, and the states carry no noise.
- **Impact:** the data is easier to learn than real robot data, which needs finite differences or filtering. Any DeLaN-vs-MLP gap may be larger or smaller under noise.
- **Action (Phase 7):** run the planned noise ablation, or else state this in the discussion as a limitation.

### D8. Row alignment differs from the PDF snippet — Low (documented)
The PDF records the state *after* `env.step` together with the previous action. We record the state *before* the step with the action applied from it (PART2 §1). The labels are equally exact; ours make s_{t+1} = RK4(s_t, u_t) hold. Mention this in the report so graders don't read it as a mistake.

### D9. The exploration policy uses nominal parameters — Low (documented)
The pumping law uses the true J and m_p l. This shapes the E distribution but is never part of an evaluated model, and the same holds for the proposed balance-pool LQR. Say so if asked in the viva.

## Phase 3 pitfalls already identified

### P1. Initialising DeLaN's damping too high — High, open
- **Evidence:** softplus(0) = 0.69. The true damping is b_x = 5×10⁻⁴ and b_θ = 2×10⁻⁶.
- **Impact:** a default-initialised D gives a damping time constant of 1.1/0.69 ≈ 1.6 s instead of 2200 s. The network would have to unlearn it from data that barely constrains it (PART1, P5), and any leftover damping biases the rollouts.
- **Action:** initialise the raw parameter at softplus⁻¹(10⁻⁴), and report the learned D next to the true values.

### P2. Fairness of the energy-drift test — Medium (by design)
- **The asymmetry:** DeLaN can switch its damping off (D = 0) for the undamped test. The MLP cannot, because it was trained on damped data.
- **Why it is negligible here:** the true damping time constants are 2200 s and 17 000 s against a 10 s test.
- **Action:** state this explicitly. If an MLP shows steady energy loss, attribute it carefully, since the true damping cannot explain more than about 0.5 % over 10 s.

## Analytical model (Phase 1)

### A1. Rail-limit rows: q̈ labels deviate by up to 2×10⁻⁴ m/s² — Low (explained)
This is the constraint solver's tolerance (10⁻⁸), not a model error. A tolerance of 10⁻¹⁵ brings it to 2×10⁻¹⁴ (PART1, check 4b). It is about 4×10⁻⁶ relative to the signal and is moot if D2 option (a) is adopted.

### A2. The models do not include the rail limit — Medium (by design)
None of the three models represents the ±1.8 m soft limit. Rollout windows that touch the limit are excluded, and MPPI keeps |x| < 1.6 m with a barrier cost. If MPPI ever reaches the rail, every model is wrong there equally.

## Environment and housekeeping

| item | status |
|---|---|
| ROS sourcing leaks into the venv through `PYTHONPATH` (pip warns about `launch-ros` needing pyyaml) | Harmless. Always run with `env -u PYTHONPATH`. |
| MuJoCo 4.0 changed `mj_fullM(m, dst, qM)` to `mj_fullM(m, d, dst)` and removed `d.qM` | Fixed. Teammates **must** use the pinned versions in `requirements.txt` (code written for 3.x will break). |
| dm_control 1.0.51 triggers 602 NumPy-2.5 DeprecationWarnings (`dm_control/mujoco/index.py:381`) | Harmless. Only that message is filtered in `pytest.ini`. |
| `__pycache__/*.pyc` was committed in the Phase 0/1 commit | Removed from the index; `.gitignore` added. |
| Disk space was low during setup (≈4 GB free) | Torch was installed without pip's cache. Keep model weights and videos small. |
| Pumping gains changed from 1.5 to 3 after a test (rail contact 2.3 % → 0.4 %) | Documented in PART2 §4. |
