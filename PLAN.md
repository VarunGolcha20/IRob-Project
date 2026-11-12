# Plan — DeLaN + MPPI swing-up of the dm_control cart-pole

## Context

The course project (IROB 2026, `IROB2026_Project_instructions (1) (1).pdf`) requires four things: a dm_control system, a physics-informed network, a black-box baseline plus the analytical model, and one closed-loop task. The approved statement (`problem_statement.txt`) is:

> Learn the forward dynamics q̈ = f(q, q̇, τ) of `cartpole/swingup` with a **Deep Lagrangian Network** (M = LLᵀ + εI, separate V(q), Coriolis terms from autograd, learned damping). Compare it with a **black-box MLP** and the **analytical Euler–Lagrange model** on one-step error, rollout RMSE (0.5 / 1 / 2 s), energy drift (10 s, unforced and undamped), data efficiency (1 / 5 / 20 episodes) and error in the out-of-range upright region. Then run the **same MPPI swing-up controller** with all three models (success over 20 seeds, time-to-balance, cart staying on the rail).

**Working style:**
- Each phase starts with a math note in `docs/PARTn_*.md`.
- Code follows, then a numeric verification against MuJoCo, then pytest checks.
- Every risk found goes into [docs/ISSUES.md](docs/ISSUES.md), with its evidence, impact and status.
- The final report shows results only. Code-generated plots are fine; the user records the videos.

**Environment:**
- venv at `~/irob_proj` (Python 3.12, dm_control 1.0.51, MuJoCo 4.0.0, torch 2.15 CPU). Pinned versions are in `requirements.txt`.
- Always run commands as `env -u PYTHONPATH ...`, because ROS's path leaks into the venv.
- The machine has 8 CPU cores and no GPU.

## Status

| phase | content | status |
|---|---|---|
| 0 | Environment, repo skeleton | **Done** |
| 1 | Analytical EL model, validated against MuJoCo | **Done**: [docs/PART1_EL.md](docs/PART1_EL.md) |
| 2 | Datasets R / E / S / U, label audit, support definition | **Done**: [docs/PART2_DATA.md](docs/PART2_DATA.md) |
| 3 | MLP + DeLaN, training | next |
| 4 | Open-loop evaluation | — |
| 5 | MPPI swing-up and balance; tuning can start now with the analytical model | — |
| 6 | Pre-evaluation demo | — |
| 7 | Ablations, report, submission, viva prep | — |

---

## 0. Physics facts the design rests on (all verified in Phase 1)

Coordinates: q = (x, θ). **θ = 0 is upright** and θ = π is hanging. τ = [10u, 0] with u ∈ [−1, 1].

| quantity | value | source / check |
|---|---|---|
| m_c, m_p, l | 1.0 kg, 0.1 kg, 0.5 m | `physics.model` |
| I_c (capsule, r = 0.045, L = 1) | 0.0094246 kg m² | hand formula = MuJoCo exactly |
| J = I_c + m_p l² | 0.0344246 kg m² | thin-rod value 0.03333 (−3.17 %) |
| b_x, b_θ | 5×10⁻⁴, 2×10⁻⁶ | damping time constants 2200 s and 17 212 s ≫ 10 s |
| h, integrator | 0.01 s, RK4 | our RK4 = MuJoCo's to 2×10⁻¹⁵ |
| rail | ±1.8 m, soft limit | `qfrc_constraint` up to 53 N in the data |
| M eigenvalues | 0.032 … 1.10, condition number 34 | ⇒ ε = 10⁻³ |
| ω (hanging), λ (upright) | 3.906 rad/s, period 1.609 s; λ = +3.906 /s, errors double every 0.177 s | ⇒ 2 s rollouts near upright amplify errors about 2400× |
| separatrix speed at the bottom | 7.81 rad/s | ⇒ the boundary between the drift groups |
| RK4 energy-drift floor, 10 s | 2×10⁻⁹ … 1.2×10⁻⁶ m_p g l; scales as h⁵ | ⇒ learned drift is meaningful only above this |

**Structural identities** (PART1 §4):
1. **M ≻ 0.**
2. **Translational symmetry:** M and V depend on θ only, and p_x is conserved.
3. **q̇ᵀ(Ṁ − 2C)q̇ = 0 for ANY M(q)** when C q̇ = Ṁ q̇ − ½ ∂_q(q̇ᵀ M q̇). Hence dH/dt = q̇ᵀ(τ − D q̇), so an untrained DeLaN already conserves its own energy.

---

## Phases 0–2 — done (summary)

- **Phase 0:** the venv, pinned requirements, the repo layout, and one model interface `model.qdd(q, qd, tau)` shared by every model, the RK4 integrator and MPPI.
- **Phase 1:**
  - [cartpole/params.py](cartpole/params.py) reads every parameter from MuJoCo.
  - [cartpole/analytical.py](cartpole/analytical.py) provides the analytical model and a thin-rod variant.
  - Seven checks pass at round-off: M, Cq̇ + g, Dq̇, q̈, `mj_inverse`, energy, and RK4 against MuJoCo.
  - The 2×10⁻⁴ q̈ residual on rail-limit rows was traced to MuJoCo's constraint-solver tolerance.
- **Phase 2:**
  - [scripts/p2_collect.py](scripts/p2_collect.py) generates the pools: R (random), E (energy pumping), S (LQR balance), U (unforced, undamped, 4 groups).
  - The splits are by episode.
  - Training uses `flatten(..., drop_limit=True)`.
  - `in_support` gives a density-based training support.
  - The random-walk theory explains why R never reaches upright: about 92 s would be needed, against 10 s episodes; measured within 3 % of theory before rail contact.
  - Fixed issues: D1, D2, D3, D5, D10 (see ISSUES.md).

**Data available for later phases:**

| pool | train / val / test (episodes) | role |
|---|---|---|
| R random | 20 / 5 / 10 | regime A; in-range test |
| E pumping | 20 / 5 / 10 | regime B; OOR-angle and OOR-velocity tests (from E_test) |
| S balance | 10 / 3 / 5 | regime B; balance-region test (S_test) |
| U unforced, undamped | 40 initial states (small / large / rotating / fast_rotating) | energy drift |

---

## Phase 3 — Models and training

**Math note:** [docs/PART3_MODELS.md](docs/PART3_MODELS.md). It covers:
- the DeLaN derivation, with Cq̇ by a forward-mode JVP;
- the proof that the energy identity holds for any M and V;
- why the inverse loss alone is degenerate, and the scale-invariant fix;
- how the loss is normalised;
- how parameters are identified from the learned M and V.

### 3.1 Dataset builder — `cartpole/dataset.py`

- **`build(regime, N, seed) -> (train, val, stats)`**
  - **Regime A:** N random episodes drawn from R_train with `rng(seed)`, without replacement.
  - **Regime B:** N R + N E + ⌈N/2⌉ S, all drawn with the same `rng`.
  - **Validation:** R_val for A; R_val + E_val + S_val for B.
  - Rail-limit rows are always dropped (ISSUES D2).
- **Inputs:** (x, sinθ, cosθ, ẋ, θ̇, τ_x). τ_θ is always 0, so feeding it would be a constant input.
- **Targets:** (ẍ, θ̈).
- **`stats`, computed on the training rows only:**
  - per-feature input mean and std (for the MLP);
  - per-axis std σ_ẍ and σ_θ̈ (for the loss);
  - std of τ_x (for the inverse loss).
- **Seeds (ISSUES D6):** 5 seeds for N = 1, 3 seeds for N = 5 and 20. Each seed changes both the episode subset and the network initialisation.

### 3.2 Architectures (matched parameter count, about 34 k each)

| | MLP (black box) | DeLaN |
|---|---|---|
| inputs | (x, sinθ, cosθ, ẋ, θ̇, τ_x), standardised | M-net and V-net see (sinθ, cosθ) only (translational symmetry, PART1 P2); q̇ and τ enter only through the EL structure |
| body | 3 × 128, softplus | L-net 2 × 128 → 3 outputs (l₁₁, l₂₁, l₂₂); V-net 2 × 128 → 1; damping D = diag(softplus(d)) |
| outputs | standardised (ẍ, θ̈), de-standardised | q̈ = M⁻¹(τ − Cq̇ − g − Dq̇); also M, V, H, τ̂ |
| parameters | 34 178 | 17 283 + 17 025 + 2 = 34 310 |

**DeLaN details, each with its reason:**
- **M = LLᵀ + εI, ε = 10⁻³, with L₁₁ = softplus(·) and L₂₂ = softplus(·).** Positive diagonals make the Cholesky factor unique and smooth. ε is 30× below λ_min = 0.032 and bounds ‖M⁻¹‖ ≤ 1000 during training.
- **Softplus everywhere.** g and C need ∂M/∂θ and ∂V/∂θ, and training backpropagates through them, so the network must be at least C². With ReLU, g and C would be piecewise constant.
- **Derivatives from one forward-mode JVP.** M and V depend on the scalar θ only, so `torch.func.jvp` with tangent e_θ gives ∂M/∂θ and ∂V/∂θ in one pass. Then:
  - Cq̇ = (∂M/∂θ) θ̇ q̇ − ½ [0, q̇ᵀ(∂M/∂θ)q̇]
  - g = [0, ∂V/∂θ]
  - q̈ is solved with the closed-form 2×2 `solve2`.

  sin and cos are computed inside the graph, so derivatives are taken with respect to θ.
- **Initialisation, without using the true parameters:**
  - Weights use the PyTorch defaults.
  - The L-diagonal biases are 0, which gives M ≈ 0.48 I. If convergence is slow, the fallback is a *data-driven* M₁₁ initialisation from least squares of τ_x on ẍ.
  - **Damping raw parameter = softplus⁻¹(10⁻⁴)** (ISSUES P1). softplus(0) = 0.69 would mean a 1.6 s damping time constant, which the data cannot undo (PART1 P5).
- **The MLP uses the same softplus activation,** so that smoothness is not a confound.

### 3.3 Loss

**Forward loss:**
$$\mathcal L_{fwd} = \tfrac1n\sum_i \Big[\big(\tfrac{\hat{\ddot x}_i - \ddot x_i}{\sigma_{\ddot x}}\big)^2 + \big(\tfrac{\hat{\ddot\theta}_i - \ddot\theta_i}{\sigma_{\ddot\theta}}\big)^2\Big]$$
It is identical for both models. Per-axis normalisation is needed because σ_θ̈ is 2–3× σ_ẍ (PART2 §2).

**Inverse loss (DeLaN only, weight λ_inv ∈ {0, 0.1, 1}, chosen on validation):**
$$\mathcal L_{inv} = \big(\tfrac{\hat\tau_x - \tau_x}{\sigma_\tau}\big)^2 + \big(\tfrac{\hat\tau_\theta}{\mathrm{sg}(\hat M_{22})\,\sigma_{\ddot\theta}}\big)^2$$
- **The degeneracy.** The unactuated row has target 0. If its residual were normalised by a fixed scale, the network could shrink it by making the θ-row terms (M₂₂, V, b_θ) smaller, which is a degenerate direction.
- **The fix.** Divide by the model's own detached M₂₂ (sg = stop-gradient). The residual then becomes an acceleration-scale quantity that cannot be reduced by rescaling.
- **The forward loss always stays in.**

### 3.4 Training settings

- **Optimiser:** Adam, lr 10⁻³ with cosine decay to 10⁻⁵, batch 512.
- **Gradient clipping:** at 10, as protection against early M⁻¹ spikes.
- **Fixed budget of 30 000 gradient steps.** Compute is then equal across N, and epochs are not comparable between N = 1 (about 950 rows) and N = 20.
- **Early stopping:** evaluate the validation loss every 500 steps and keep the best weights; patience is 10 evaluations.
- **Precision:** float32 for training. All evaluations in Phase 4 run in float64, so round-off cannot pollute the drift.
- **Runs:**
  - regime A: (5 + 3 + 3) seeds × 2 models = 22;
  - regime B, N = 20: 3 seeds × 2 models = 6;
  - optional regime B, N = 5: 6.
  - About 30–35 runs in total, at roughly 1–3 min each on CPU.
- **Parallelism:** 4 processes with `torch.set_num_threads(2)` each.
- **Saved per run:** `weights/{model}_{A|B}{N}_s{seed}.pt`, holding the state dict, the config, the normalisation stats, the best validation loss and the step count.

### 3.5 Tests — `tests/test_models.py`

1. A *random-init* DeLaN gives symmetric M with eigenvalues ≥ ε for 10⁴ random θ.
2. A *random-init* DeLaN satisfies q̇ᵀ(Ṁ − 2C)q̇ = 0 to 10⁻¹⁰ in float64. With τ = 0 and D = 0, its own H along a 10 s RK4 rollout stays constant to the integrator floor. This shows the guarantee comes from structure.
3. Plugging the analytical M and V into the DeLaN pipeline reproduces `AnalyticalModel.qdd` to 10⁻¹². This checks the JVP-based Coriolis code.
4. DeLaN inverse ∘ forward is the identity, i.e. τ̂(q, q̇, q̈(τ)) = τ.
5. Both models overfit one batch of 512 rows to a loss below 10⁻⁴. This is a plumbing check.
6. **`torch.func.jvp` works under `torch.no_grad()` for MPPI inference.** If it doesn't, MPPI wraps the DeLaN call in `torch.enable_grad()`.

### 3.6 Outputs and acceptance

**Outputs:**
- `results/p3_training.json`;
- `figures/p3_training_curves.*`;
- **`figures/p3_learned_structure.*`:** M₁₁, M₁₂, M₂₂ and V(θ) − V(π) against the truth over θ ∈ [−π, π], for regime A20 and B20, with the training support shaded. This is the key DeLaN figure.

**Parameter identification, with two independent estimates of m_p l:**
- Least-squares fit of the learned M₁₁ → m_c + m_p.
- Fit of M₁₂ on cosθ → m_p l (**inertial**).
- Fit of M₂₂ → J.
- Fit of V on cosθ → m_p g l (**gravitational**).
- **Consistency check:** m_p l from M₁₂ against m_p g l / g from V. Agreement is a physics check the MLP cannot offer. Report it with D against the true values.

**Acceptance criteria:**
- tests 1–6 pass;
- validation loss converges for every run;
- in regime A20, the in-range one-step nRMSE is a few percent or less for both models;
- DeLaN's identified parameters lie within a few percent of the truth *inside* the training support.

---

## Phase 4 — Open-loop evaluation

**Math note:** [docs/PART4_EVAL.md](docs/PART4_EVAL.md). **Script:** `scripts/p4_eval.py`, which writes `results/p4_*.json` and `figures/p4_*`.

**Models evaluated**, all through the same `rk4_step` in float64:
- Analytical (oracle);
- Analytical (thin rod);
- Analytical with +1 % on every parameter (a reference curve);
- MLP and DeLaN for every regime, N and seed.

| metric | definition | why this definition | ISSUES |
|---|---|---|---|
| One-step | nRMSE of q̈ per axis (normalised by test std), plus the state error after one RK4 step; **per region**: hanging (cosθ < −0.8), mid, upright (cosθ > 0.8), balance, OOR-angle, OOR-velocity, density-supported vs not | Overall averages hide region-wise failures because hanging dominates the data. | D4 |
| Rollout RMSE at 0.5 / 1 / 2 s | Open-loop replay of the recorded u from start points every 25 steps on R_test, E_test and S_test. RMSE of wrapped θ and of x at each horizon; median and IQR across windows; **windows that touch the limit are excluded** | Replay keeps plant inputs identical. Wrapping avoids fake 2π errors. The median resists the exponential divergence near upright (λ = 3.9 /s). | A2 |
| Valid-prediction time | First t with \|Δθ\| > 0.1 rad, averaged over windows | A single number that degrades gracefully, unlike RMSE after divergence. | — |
| Energy drift (10 s) | U pool, τ = 0. Report max_t \|H_true(ŝ(t)) − H_true(s₀)\| / (m_p g l) **per group, with the group's support label**. DeLaN is run twice, with D = 0 and with its learned D. Also DeLaN's own H(t), the p_x(t) drift, and the state error. The analytical RK4 and MuJoCo give the floor. | H_true along the predicted trajectory is the only fair energy score for the MLP. The D = 0 asymmetry is negligible here (damping time constants ≫ 10 s) and is stated. | D5, P2 |
| Data efficiency | Every metric above against N = 1, 5, 20 (regime A), mean ± std over seeds, on log-log axes | Required by the brief, with three sizes. | D6 |
| OOR | One-step and 0.5 s rollout error on OOR-angle and OOR-velocity, **error heat map over (θ, θ̇) with the density-support contour**, and **error against distance to support** (Chebyshev distance in bins to the nearest supported bin) | Shows where extrapolation fails, and measures distance from what was *actually* trained on, not from a max-based box. | D1 |
| Inverse dynamics | DeLaN τ̂ against `mj_inverse` on test rows; the row-2 residual (should be ≈ 0) | Comes free with DeLaN; the MLP cannot do it. | — |
| Identified parameters | From Phase 3.6, compared with MuJoCo | Required "where applicable". | — |
| Compute | µs per batched q̈ evaluation (batch 512), per model | Feeds the MPPI cost discussion. | — |

**Expected findings, written down before running so they can be checked honestly:**
- DeLaN beats the MLP on energy drift (by structure) and on OOR-velocity (exact quadratic dependence on q̇).
- On OOR-angle the outcome is genuinely open, because both models must extrapolate M and V or f in θ.
- Both models degrade on 2 s rollouts near upright, as the λ-amplification predicts.
- DeLaN's margin should be largest at N = 1.

---

## Phase 5 — MPPI swing-up and balance

**Math note:** [docs/PART5_MPPI.md](docs/PART5_MPPI.md). **Code:** `cartpole/mppi.py`, `scripts/p5_mppi.py`. Tuning starts **now** with the analytical model, in parallel with Phase 3.

### 5.1 Algorithm (information-theoretic MPPI, Williams et al. 2017)

**One iteration:**
1. Sample K noise sequences ε_k ∈ ℝ^H, with time correlation from an AR(1) filter at coefficient 0.8, for smooth forces.
2. Form U_k = clip(U + ε_k, −1, 1).
3. Roll out all K in one batched call: RK4 with Δt_plan = 0.02 and τ = [10u, 0].
4. Compute the cost S_k = Σ_t c(s_t, u_t) + φ(s_H) + λ Σ_t u_tᵀ σ⁻² ε_t.
5. Weight w_k = softmax(−(S_k − min S)/λ) and update U ← U + Σ_k w_k ε_k.
6. Apply U₀, shift U by one step (warm start), and repeat at 100 Hz.

**Temperature:** λ is chosen by bisection each step so that ESS = 1/Σw² sits in [0.1 K, 0.2 K]. This is a statistical target, not a hand-tuned temperature.

**Starting settings and reasons:**

| setting | value | reason |
|---|---|---|
| planning step and horizon | Δt_plan = 0.02 s, H = 75 (1.5 s ≈ one period) | ωΔt = 0.08, so RK4 error is negligible. One period of look-ahead shows the payoff of a pump. Allowed because the models are continuous-time. |
| samples | K = 512 | |
| noise | σ = 0.5 (5 N) | |
| stage cost | Bryson-style weights: w_θ(1 − cosθ), with (1 − cosθ) ∈ [0, 2]; x²/1²; ẋ²/2²; θ̇²/5²; 0.1 u² | Each term is O(1) at its acceptable deviation. |
| rail barrier | 10³ · relu(\|x\| − 1.6)² | 0.2 m of margin to a limit that no model knows (ISSUES A2). |
| terminal cost | 10 × the stage cost | |
| energy shaping (if needed) | add w_E (E_p − m_p g l)² if MPPI stalls at the bottom | E_p uses the known geometry, identical for all models, so it is part of the task. |

**Fallback:** MPPI for swing-up, then LQR when |θ| < 0.3 and |θ̇| < 2, with hysteresis. The LQR gain comes from the **same model's** autograd Jacobian at upright, so every controller depends only on the model under test. The Bryson weights are the same as for pool S.

### 5.2 Tuning protocol (no test-seed leakage)

- Tune **only** with the analytical model on seeds 100–109.
- Log every configuration tried to `results/p5_tuning.csv`, so no result is cherry-picked.
- **Freeze** the chosen configuration to `results/p5_mppi_config.json`.
- **Acceptance:** at least 18/20 success with the analytical model on seeds 0–19. Otherwise the problem is the controller, not the models.

### 5.3 Evaluation

**Models:** analytical, thin-rod, MLP-B20, DeLaN-B20. For the learned models, use the training seed with the best *validation* loss; all 3 seeds if time permits.

**Stress test:** MLP-A20 and DeLaN-A20, which have never seen upright.

**Runs:** env seeds 0–19 per model. The seed sets the initial state and the MPPI noise.

**Metrics:**
- **Balanced:** |wrap θ| < 0.2 and |θ̇| < 1.
- **Time-to-balance:** the first t at which "balanced" then holds for 2 s.
- **Success:** balanced within the 10 s episode **and** the rail limit never active.
- **Also:** max |x|, dm_control return, ms per control step, and the number of control steps where the MPPI/LQR switch occurs (if the fallback is used).

**Statistics:**
- **Wilson 95 % interval** on each success rate. With n = 20, 18/20 gives [0.70, 0.97], so claim only differences the intervals support.
- A cumulative "fraction balanced against time" curve per model (Kaplan–Meier style, where failures are censored at 10 s).

**Compute:**
- About 0.1 s per step × 1000 steps × 20 seeds × 6 models ≈ 3.3 CPU-hours.
- `multiprocessing` over seeds brings this under 1 hour.

**Outputs:**
- `results/p5_closed_loop.json`;
- `figures/p5_success.*` (bar chart with Wilson intervals);
- `figures/p5_time_to_balance.*`;
- `figures/p5_trajectories.*` (θ, x, u against t for one seed per model).

---

## Phase 6 — Pre-evaluation demo

**`scripts/demo.py --part {rollout,swingup} --model {analytical,mlp,delan}`:**
- **rollout** plots a learned rollout over the MuJoCo trajectory for one test episode, live.
- **swingup** runs MPPI in dm_control, with the optional interactive viewer `dm_control.viewer`.

**Slides (4):**
1. The problem and the Phase 1 validation.
2. The DeLaN structure, with learned M and V against the truth.
3. Rollout, energy-drift and OOR plots.
4. Swing-up success table, plus a live run.

The live demo is 20 % of the grade, so it must run from one command each. Freeze those commands in the README.

---

## Phase 7 — Ablations, report, submission, viva

**Ablations,** kept only if time allows (each linked to an issue or to a claim in the report):

| ablation | question it answers | ISSUES |
|---|---|---|
| DeLaN with x as an input to M and V | What does the translational-symmetry prior buy? | — |
| λ_inv ∈ {0, 0.1, 1} | Does the inverse (EL-residual) loss help? | — |
| Inverse-density-weighted loss | Does reweighting away from hanging help either model? | D4 |
| Measurement noise (σ_q = 10⁻³, σ_q̇ = 10⁻²; q̈ from finite differences) | Does the DeLaN advantage survive realistic labels? | D7 |
| Closed loop at N = 5 (regime B) | Data efficiency in closed loop | — |

**Videos:**
- `scripts/render.py` writes MP4s: side-by-side swing-ups (analytical | MLP | DeLaN, same seed), and a model rollout overlaid on MuJoCo.
- The user curates the final demo video.

**Report (PDF; Introduction / Problem Statement / Methodology / Results / Conclusion),** mapped to the instruction items:

| item | content |
|---|---|
| 1. Problem statement | §0 + PART1 §1 |
| 2. Analytical model | PART1; figures p1_* |
| 3. Data | PART2; figure p2_coverage; split table; OOR and support definitions; the row-alignment note (D8) and the nominal-parameter note (D9) |
| 4. Models | PART3; architecture table, losses, seeds; figure p3_learned_structure |
| 5. Evaluation | PART4; figures p4_* |
| 6. Closed loop | PART5; figures p5_* |
| 7. Discussion | where the prior helps (energy, OOR-velocity, low N) and where it doesn't (θ-extrapolation, upright instability); compute cost; limitations (D7, A2); links to A2 (EL derivation, parameter gaps, integrator order) and A3 (model-based control) |

**Submission (`Robotics_project_group#no.zip`):**
- the report PDF;
- the code, `requirements.txt`, and a README with one command per figure;
- `weights/`;
- `minutes/`;
- the video or a link to it.

**Viva prep:** every member must be able to derive on a whiteboard:
- M, C and g;
- Ṁ − 2C skew-symmetric, and why the energy identity holds for any learned M and V;
- the reasons for ε, softplus and Cholesky;
- the inverse-loss degeneracy and its fix;
- the random-walk argument for the dataset;
- the MPPI weight derivation;
- the Wilson interval.

The `docs/PART*.md` notes and ISSUES.md double as the study guide.

---

## Questions for the instructor meeting (book it before the meeting-request cutoff)

1. Is it acceptable that DeLaN's M and V take θ only (a translational-symmetry prior) while the MLP gets the full input?
2. Is it acceptable to drop rail-limit rows from training (ISSUES D2)?
3. Are the extra balance pool S (controller rollouts) and the density-based OOR definitions acceptable?
4. Is the success definition (|θ| < 0.2 held for 2 s, rail limit never active) acceptable?

## Risks

| risk | mitigation |
|---|---|
| MPPI fails even with the analytical model | Tune early; ESS-adaptive λ; energy-shaping terminal cost; MPPI + LQR fallback |
| DeLaN training unstable | ε floor, gradient clipping, damping initialised near 0, float64 retry, data-driven M₁₁ initialisation |
| `torch.func.jvp` slow or blocked under no_grad in MPPI | Phase 3 test 6; `enable_grad` wrapper; the closed-form 2×2 algebra keeps the cost low |
| MLP diverges in rollouts | This is a result. First rule out normalisation bugs using the training-set one-step error. |
| Closed-loop CPU time | Δt_plan = 0.02; batched torch; multiprocessing over seeds |
| Teammates on other library versions | Pinned `requirements.txt`; the MuJoCo 4.0 API change is documented |
