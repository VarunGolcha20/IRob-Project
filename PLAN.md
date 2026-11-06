# Plan — DeLaN + MPPI swing-up of the dm_control cart-pole

## Context

The course project (IROB 2026, `project/IROB2026_Project_instructions (1) (1).pdf`) requires four things: a dm_control system, a physics-informed network, a black-box baseline plus the analytical model, and one closed-loop task. The approved statement (`project/problem_statement.txt`) is:

> Learn the forward dynamics q̈ = f(q, q̇, τ) of `cartpole/swingup` with a **Deep Lagrangian Network** (M = LLᵀ + εI, separate V(q), Coriolis terms from autograd, learned damping). Compare it with a **black-box MLP** and the **analytical Euler–Lagrange model** on one-step error, rollout RMSE (0.5 / 1 / 2 s), energy drift (10 s, unforced and undamped), data efficiency (1 / 5 / 20 episodes) and error in the out-of-range upright region. Then run the **same MPPI swing-up controller** with all three models (success over 20 seeds, time-to-balance, cart staying on the rail).

The project folder holds only the PDF and the statement. There is no code yet.

**Environment facts (checked on this machine):** Python 3.12.3, which dm_control supports (3.10–3.12). NumPy 1.26 and SciPy 1.11 are installed. dm_control, mujoco and torch are not. There is no NVIDIA GPU; the machine has 8 CPU cores and 7 GB of RAM. `python3-venv` is missing, but `virtualenv` is already in `~/.local`, so no sudo is needed.

**Working style (carried over from A2):** each phase starts with a short math note in `docs/`. Code follows, then a numeric verification against MuJoCo, before the next phase starts. The final report shows results only, with no detours. I write the code and the user records the videos.

**Phase order (course milestones):** env + analytical model + data first; networks trained next; then a pre-evaluation demo; then the final report, code and viva.

---

## 0. The physics we build on (the basis for every assumption)

**Coordinates.** q = (x, θ). x is the slider (cart) position. θ is the hinge angle, with **θ = 0 upright** because the pole capsule runs `fromto 0 0 0 → 0 0 1`, and **θ = π hanging**. Swing-up episodes start at θ ≈ π. The hinge axis is +y, so the centre of mass (COM) of the pole sits at (x + l sinθ, l cosθ).

**Actuation.** Only the cart is driven: τ = [10·u, 0] with u ∈ [−1, 1] (motor gear 10). MuJoCo's `qfrc_actuator` already holds this vector, so underactuation enters the equations as a generalized force with a zero second row. No special handling is needed.

**Parameters to read from `physics.model`.** We never assume these values; we read them from the model. The values below are what I expect, and Phase 1 checks every one.

| symbol | meaning | expected | source |
|---|---|---|---|
| m_c | cart mass | 1.0 kg | `body_mass[cart]` |
| m_p | pole mass | 0.1 kg | `body_mass[pole_1]` |
| l | hinge → pole COM | 0.5 m | `body_ipos` |
| I_c | pole inertia about COM (y axis) | ≈ 0.0094 kg m² (cylinder + 2 hemispheres, r = 0.045) | `body_inertia` |
| J = I_c + m_p l² | pole inertia about hinge | ≈ 0.034 kg m² | derived |
| b_x, b_θ | viscous damping | ≈ 5e-4, ≈ 2e-6 | `dof_damping` |
| armature, frictionloss | — | 0, 0 (to verify) | `dof_armature`, `dof_frictionloss` |
| h | timestep / integrator | 0.01 s, RK4 | `opt.timestep`, `opt.integrator` |
| rail | slider range | ±1.8 m (soft limit) | `jnt_range` |

**Euler–Lagrange model** (derived from T and V):

- T = ½(m_c+m_p)ẋ² + m_p l cosθ ẋθ̇ + ½Jθ̇²
- V = m_p g l cosθ
- M(θ) = [[m_c+m_p, m_p l cosθ], [m_p l cosθ, J]]
- C(q,q̇)q̇ = [−m_p l sinθ θ̇², 0]
- g(q) = ∂V/∂q = [0, −m_p g l sinθ]
- D = diag(b_x, b_θ)
- M q̈ + C q̇ + g + D q̇ = τ

**Physics facts that set the design choices** (each is worked out in Phase 1):

1. **Translational symmetry.** L does not depend on x, so x is a cyclic coordinate. Noether's theorem then says p_x = ∂L/∂ẋ is conserved when τ = 0, and M and V depend on θ only. → In DeLaN, the M and V networks take (sinθ, cosθ) only. This is a symmetry prior, not parameter knowledge. An ablation keeps x as an input.
2. **Scale of M.** At cosθ = ±1, M has λ_min ≈ 0.032 and λ_max ≈ 1.10, so the condition number is about 34. → Use **ε = 1e-3**. That is 30 times smaller than λ_min, so it adds no bias, and it bounds ‖M⁻¹‖ ≤ 1000, which prevents blow-ups during training.
3. **Time scales.** Hanging, the linearised frequency is ω² = m_p g l (m_c+m_p) / (J(m_c+m_p) − (m_p l)²). That gives ω ≈ 3.9 rad/s and a period of about 1.6 s. Upright, there is an unstable eigenvalue of the same size, λ ≈ +3.9 /s, so errors double every ≈ 0.18 s and grow about 2400× in 2 s. → 2 s rollouts that pass near upright are expected to diverge for *every* imperfect model. We interpret them against a reference curve: the analytical model with 1 % parameter error.
4. **Damping is negligible.** The time constants are (m_c+m_p)/b_x ≈ 2200 s and J/b_θ ≈ 17000 s, both far longer than a 10 s episode. → The learned D is practically unidentifiable. We report it anyway, and we note that it is negligible for the MLP's drift test.
5. **Energy balance.** Ṁ − 2C is skew-symmetric, so dH/dt = q̇ᵀτ − q̇ᵀDq̇. → When τ = 0 and D = 0, DeLaN conserves its own energy *by construction*, in continuous time. The only drift left comes from RK4, which is O(h⁴). The MLP has no such guarantee.
6. **Exact labels.** After `env.step`, calling `physics.forward()` recomputes `qacc` at the new state with the same ctrl. q̈ is therefore the exact continuous-time acceleration, not a finite difference. → We train on accelerations directly (no trajectory loss is needed), and the learned models are continuous-time, so MPPI can integrate them with any step size.
7. **Rail limit.** The joint limit is a soft constraint. When the limit is active, `qfrc_constraint ≠ 0` and the Lagrangian alone no longer explains the motion. iid random forcing at ±10 N makes the cart's velocity random-walk to about 1.7 m/s std over 10 s, so it hits the limit often. → Record `qfrc_constraint`. Training uses **τ_gen = qfrc_actuator + qfrc_constraint**, because constraint forces are just more generalized forces in the Lagrange equations. Rollout test windows exclude any window in which the limit becomes active. MPPI keeps |x| < 1.6 m.

---

## Phase 0 — Environment and repo skeleton

- Run `python3 -m virtualenv ~/irob_proj && source ~/irob_proj/bin/activate`.
- Run `pip install dm_control mujoco numpy scipy matplotlib pandas tqdm imageio[ffmpeg]` and `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The CPU wheel avoids roughly 2 GB of CUDA packages, which this machine can't use anyway.
- Smoke test: run the PDF's random-episode script and print the shape `(1000, 8)`. Set `MUJOCO_GL=egl` for headless rendering, with `osmesa` as the fallback.
- Freeze `requirements.txt`.
- Repo layout (all under `project/`):
  ```
  cartpole/            # python package
    params.py          # read every parameter from physics.model (no hard-coded numbers)
    analytical.py      # EL model in numpy + batched torch (same f(q,q̇,τ) interface)
    integrators.py     # batched RK4 (torch), wrapped-angle helpers
    data.py            # collectors: random, energy-pumping, unforced-undamped
    models/mlp.py, models/delan.py
    train.py, evaluate.py, mppi.py, closed_loop.py, plotting.py, render.py
  scripts/             # p1_validate.py, p2_collect.py, p3_train.py, p4_eval.py, p5_mppi.py ...
  docs/                # PART1_EL.md, PART2_DATA.md, PART3_MODELS.md, PART4_EVAL.md, PART5_MPPI.md
  data/ weights/ results/ figures/ report/ minutes/
  tests/               # pytest: the structural checks listed in each phase
  README.md            # one command per figure (required by submission)
  ```
- **One common interface:** `model.qdd(q, qd, tau) -> qdd`, batched, in torch. All three models (analytical, MLP, DeLaN) implement it and share **the same RK4 integrator and the same MPPI**. The comparison isolates the model because nothing else differs.
- Start `minutes/` now with one file per meeting. Minutes are a required deliverable.

## Phase 1 — Analytical model and validation

**Math (`docs/PART1_EL.md`):**
- Derive T, V, M, C, g and D as in §0, with the capsule inertia derived by hand (cylinder plus two hemispheres, parallel-axis theorem).
- Prove that Ṁ − 2C is skew-symmetric and derive the energy balance dH/dt = q̇ᵀ(τ − Dq̇).
- Derive the hanging and upright linearisations: eigenvalues ±iω and ±λ.
- Derive p_x conservation (Noether).

**Code:** `params.py` and `analytical.py`.

**Validation.** All checks run along recorded random trajectories and must pass to round-off (≲1e-10):
1. M(q) against `mujoco.mj_fullM`.
2. Cq̇ + g against `qfrc_bias`.
3. −Dq̇ against `qfrc_passive`.
4. Forward dynamics q̈_analytical against `qacc`.
5. Inverse dynamics τ̂ = Mq̈ + Cq̇ + g + Dq̇ against `mujoco.mj_inverse → qfrc_inverse`, as the instructions require, with `qfrc_constraint` accounted for on limit rows.
6. Our torch RK4 with dt = 0.01, rolled out against MuJoCo's RK4 for 10 s without limit hits. Agreement should be about 1e-10. This shows that our integrator *is* MuJoCo's, so integrator mismatch is gone as an error source. That was the lesson from A2.
7. The hand-derived capsule I_c against `body_inertia`.

**Extra baseline, "Analytical (thin-rod)":** use I = m_p L²/3 (L = 1 m), as a textbook / A2-style derivation would. J is then about 3 % low. This gives a realistic "hand-derived model with structural error" next to the exact oracle. It also connects to A2, which found URDF-vs-MuJoCo parameter gaps.

**Deliverable figure:** residual histograms for checks 1–5, plus a table of parameters with their MuJoCo values.

## Phase 2 — Data generation → milestone: env, analytical model and data ready

**Row format.** Each row holds (x, θ, ẋ, θ̇, u, τ_act, τ_constraint, q̈, qfrc_passive, energy[2], limit_active), recorded with the PDF's `physics.forward()` pattern at 100 Hz. Each episode is 1000 steps (10 s), the dm_control default.

**Pools.** Episodes are split by **episode, not by row**. Consecutive 100 Hz rows are almost identical, so a row-level split would leak test data into training.

| pool | excitation | episodes | purpose |
|---|---|---|---|
| **R** random | u ~ U[−1,1] iid each step, as in the statement and PDF | 20 train, 5 val, 10 test | the "in-range" regime |
| **E** energy-pumping + noise | see below | 20 train, 5 val, 10 test | upright / high-energy coverage; held-out out-of-range (OOR) test |
| **U** unforced, undamped | `dof_damping[:] = 0`, u = 0, energy flag on | 30 initial states × 10 s | energy-drift ground truth |

**Why R alone misses the upright states.** iid noise at 100 Hz spreads its power evenly over 0–50 Hz. Only a tiny fraction lands near the 0.62 Hz pendulum resonance, so θ stays near π. We confirm this by reporting the θ histogram and the maximum of cosθ in R.

**Energy-pumping policy (Åström–Furuta).** Let E_p = ½Jθ̇² + m_p g l cosθ. From the pole equation, dE_p/dt = −m_p l cosθ θ̇ ẍ. Choosing ẍ_des = k(E_p − E_ref) θ̇ cosθ gives V = ½(E_p − E_ref)² a non-positive derivative, so the energy converges to E_ref.
- Add the cart-centering term −k_x x − k_v ẋ.
- Convert to an action with u = (m_c+m_p) ẍ_des / 10, plus Ornstein–Uhlenbeck noise (correlation time ≈ 0.2 s, near resonance), then clip.
- Draw E_ref per episode from U[−m_p g l, 1.5 m_p g l]. That range covers small swings, the separatrix, and full rotations, so trajectories sweep the whole (θ, θ̇) phase plane.
- The noise also breaks the θ̇ cosθ = 0 deadlock at rest.
- The policy uses nominal parameters. That is acceptable because it only collects data and is never part of an evaluated model.

**Test regions, defined from the data rather than chosen by hand:**
- **In-range:** test episodes from R.
- **OOR-angle:** states in E-test with cosθ > 0 (pole above horizontal). We check that R has zero support there.
- **OOR-velocity:** E-test states that are hanging (cosθ < −0.8) but have |θ̇| > max|θ̇| in R-train. This separates two kinds of extrapolation. DeLaN is *exactly* quadratic in q̇ and linear in τ, so it should extrapolate in velocity for free. In angle, its M and V networks must extrapolate on their own. That is an honest account of where the structure helps.

**Training regimes:**
- **A, random only:** N ∈ {1, 5, 20} R episodes. Used for one-step error, rollouts, drift, OOR and data efficiency.
- **B, random + pumping:** N R + N E episodes. N = 20 is the main model for MPPI, so every model has seen upright states and the comparison is fair. N = 5 is optional, for closed-loop data efficiency.

**Measurement noise:** none in the main study, because MuJoCo gives exact states and q̈. One optional robustness run adds σ_q = 1e-3 and σ_q̇ = 1e-2 to the inputs.

**Deliverable:** a phase-portrait coverage figure (θ vs θ̇ for R and E, with the OOR regions shaded) and a dataset-size table.

## Phase 3 — Models and training → milestone: networks trained

**DeLaN (`models/delan.py`):**
- **L-net:** (sinθ, cosθ) → 2×64 → 3 outputs (l₁₁, l₂₁, l₂₂). The diagonal goes through softplus, so L is a valid Cholesky factor. Then M = LLᵀ + εI with ε = 1e-3 (§0.2). The diagonal biases start at log-scales matched to M₁₁ ~ 1 and M₂₂ ~ 0.03. The two entries differ by 30×, and starting near the right scale avoids a long transient.
- **V-net:** (sinθ, cosθ) → 2×64 → 1.
- **Damping:** D = diag(softplus(d)), which guarantees q̇ᵀDq̇ ≥ 0, i.e. passivity.
- **Activation: softplus**, smooth (C^∞). Gravity needs ∂V/∂θ and Coriolis needs ∂M/∂θ, and training backpropagates through these derivatives, so the network needs at least C². With ReLU, g and C would be piecewise constant and discontinuous.
- **Efficient derivatives.** M and V depend on the scalar θ only, so **one forward-mode JVP** (`torch.func.jvp`, tangent e_θ) gives both ∂M/∂θ and ∂V/∂θ. Then:
  - Cq̇ = (∂M/∂θ) θ̇ q̇ − ½[0, q̇ᵀ(∂M/∂θ)q̇]
  - g = [0, ∂V/∂θ]
  - q̈ = solve(M, τ − Cq̇ − g − Dq̇), using a closed-form 2×2 solve.

  The same code serves training (with `create_graph`) and MPPI inference.
- Angles go into sin/cos *inside* the graph, so derivatives are taken with respect to θ, not with respect to the features. This handles wrap-around and multi-turn θ.
- Outputs include M(θ), V(θ), H = ½q̇ᵀMq̇ + V, and inverse dynamics τ̂.

**MLP baseline (`models/mlp.py`):** (x, sinθ, cosθ, ẋ, θ̇, τ₁) → 3×128 softplus → (ẍ, θ̈). Inputs and outputs are standardised. Its parameter count roughly matches DeLaN's, and it uses the same optimiser, data, normalisation, early stopping and seeds. The only difference is the structure.

**Loss:**
- Forward term: Σ‖(q̈̂ − q̈)/σ_q̈‖². Each axis is normalised by its std because θ̈ (~30 rad/s²) would otherwise swamp ẍ (~9 m/s²).
- Plus λ_inv · ‖(τ̂ − τ)/σ_τ‖², the Euler–Lagrange residual as in Lutter et al., with λ_inv ∈ {0, 0.1, 1} picked on the validation set.
- **The forward term must stay.** On the unactuated row τ₂ = 0, so the inverse residual alone has a degenerate direction: the θ-row terms (M₂₂, V, b_θ) shrink toward 0 and the residual vanishes. The forward loss pins them down.
- The second row cannot be rescaled arbitrarily because M must be symmetric and M₁₂ = m_p l cosθ ≠ 0 couples it to the actuated row. That coupling is why DeLaN can identify M and V in physical units.

**Training:** Adam at 1e-3 with cosine decay, batch 512, early stopping on validation episodes, gradient clipping at 10. Float32 for training, **float64 for every evaluation rollout**, so round-off does not pollute drift numbers. Use 3 seeds per configuration, the minimum needed to show spread: 2 models × 3 sizes × 2 regimes × 3 seeds = 36 short CPU runs.

**Structural unit tests (`tests/`), which also make good viva demonstrations:**
1. With a *randomly initialised* DeLaN, M is symmetric with eig ≥ ε for any θ.
2. With a *random* DeLaN, q̇ᵀ(Ṁ − 2C)q̇ ≈ 0, and H stays constant along a rollout with τ = 0 and D = 0, up to O(h⁴). This shows the guarantee comes from structure, not training.
3. Plugging the analytical M and V into the DeLaN autograd pipeline reproduces `analytical.qdd` to 1e-12. This checks the Coriolis-by-autograd code.

**Deliverable figure: learned vs true M₁₁, M₁₂, M₂₂ and V over θ ∈ [−π, π]**, with the training support shaded. Least-squares fits of the learned curves on the analytical basis {1, cosθ} give the identified m_c+m_p, m_p l, J and m_p g l, which we compare with MuJoCo. This covers the "identified parameters" item in the instructions.

## Phase 4 — Open-loop evaluation

All three models (plus thin-rod) use the same RK4 at dt = 0.01 and the same recorded inputs.

| metric | definition | why this definition |
|---|---|---|
| One-step error | nRMSE of q̈ per axis; also the state error after one RK4 step | q̈ is the training target. The state error is what MPPI actually feels. |
| Rollout RMSE at 0.5 / 1 / 2 s | Replay the recorded u open-loop from ≥200 start points per test set. Report RMSE of wrapped θ and of x; median and IQR across windows. Windows with limit activity are excluded. | The open-loop replay keeps the plant inputs identical. Wrapped angles prevent fake 2π errors. The median resists the exponential divergence near upright (§0.3). |
| Energy drift (10 s) | From pool U, roll out with τ = 0 (and D = 0 for DeLaN, which its structure allows). Plot \|H_true(ŝ(t)) − H_true(s₀)\| / (m_p g l), using MuJoCo's `data.energy` as ground truth. Also report DeLaN's own learned H(t), and p_x(t) drift as a second invariant. | H_true is evaluated *along each model's predicted trajectory*, which is the only fair way to score the MLP since it has no energy. Normalising by m_p g l (≈ 0.49 J) is the natural energy scale. The MuJoCo-RK4 and analytical-RK4 drift give the integrator floor. |
| Data efficiency | Every metric above for N = 1, 5, 20 episodes (regime A), mean ± std over 3 seeds | Required: at least 3 sizes. |
| OOR error | One-step and 0.5 s rollout error on OOR-angle and OOR-velocity, plus an error heat-map over the (θ, θ̇) plane with the training-support contour | Shows *where* extrapolation fails rather than a single number. |
| Inverse dynamics (DeLaN and analytical) | τ̂ against `mj_inverse`; the row-2 residual should be ≈ 0 | Comes free with DeLaN. The MLP cannot produce it. |
| Compute | µs per batched q̈ evaluation | Feeds the MPPI cost discussion. |

Reference curve: the analytical model with all parameters perturbed by +1 %, shown so the size of each learned model's error has a physical meaning.

## Phase 5 — MPPI swing-up and balance

**Start Phase 5 with the analytical model as soon as Phase 1 passes, in parallel with Phases 2–3.** MPPI needs only `qdd`. If MPPI fails with the *exact* model, the problem is tuning, not learning. This removes the "about a day of MPPI tuning" risk from the critical path.

**Algorithm.** Information-theoretic MPPI (Williams et al. 2017):
- Sample K control sequences U + ε_k, with ε_k ~ N(0, σ²I).
- Roll them out in one batched torch call.
- Compute the cost S_k = Σ c(s_t, u_t) + φ(s_H) + λ Σ uᵀσ⁻²ε.
- Weight each sequence by w_k ∝ exp(−(S_k − min S)/λ) and update U ← U + Σ w_k ε_k.
- Warm-start by shifting U one step.

**Settings, each with a reason:**
- **Planning step and horizon:** Δt_plan = 0.02 s with RK4 on the continuous-time model, horizon H = 75 (1.5 s, about one pendulum period). This is allowed because the model is continuous-time (§0.6). ωΔt ≈ 0.08, so RK4 error is negligible. A full period of look-ahead is enough to see the payoff of a pump.
- **Samples:** K = 512 to start.
- **Noise:** σ = 0.5 in action units (5 N), with time-correlated noise or smoothing for smooth forces.
- **Temperature:** λ is adapted each step so that the effective sample size, ESS = 1/Σw², stays at 10–20 % of K. This replaces a hand-tuned temperature with a statistical target.
- **Replanning:** apply the first action and replan at 100 Hz, the env rate. If CPU time is too high, use a zero-order hold of 2 steps.
- **Stage cost:** c = w_θ(1 − cosθ) + w_x x² + w_θ̇ θ̇² + w_ẋ ẋ² + w_u u², plus a steep barrier on |x| > 1.6 m. The 1.6 m threshold leaves 0.2 m of margin to the ±1.8 limit, which no model represents.
- **Terminal cost:** φ uses the same terms, more heavily weighted. If MPPI stalls at the bottom because the cost is myopic, add an **energy-shaping term** w_E(E_p − m_p g l)². This uses known task geometry and is identical for all models, so it is part of the task, not the model.
- **Fallback if balancing is jittery:** MPPI for swing-up, then LQR when |θ| < 0.3. The LQR gain comes from the *same model's* autograd Jacobian at the upright, so every controller still depends only on the model being tested.

**Fairness protocol:**
- Tune all hyperparameters with the analytical model on tuning seeds 100–109, then **freeze them**.
- Evaluate all three models (plus thin-rod) on seeds 0–19, which set both the env initial state and the MPPI noise.
- There is no per-model tuning.

**Closed-loop metrics:**
- **Balanced:** |wrap θ| < 0.2 rad (≈11°, where sinθ ≈ θ to within 0.7 %, i.e. the linear regime) and |θ̇| < 1 rad/s.
- **Time-to-balance:** the first t at which "balanced" holds continuously for 2 s. Two seconds is longer than one natural period, which rules out a pass through the upright counting as balance.
- **Success:** balanced within the 10 s episode, and the rail limit never active.
- **Also reported:** max |x|, whether `qfrc_constraint` ever became non-zero, dm_control episode return, and ms per control step.
- **Statistics:** success rate with a **Wilson 95 % interval**. With n = 20, for example, 18/20 gives [0.70, 0.97], so the report only claims differences the intervals support.

**Process:**
- Run in parallel over seeds with `multiprocessing` (8 cores).
- Budget: about 0.1 s per control step × 1000 steps × 20 seeds × 4 models ≈ 2–3 CPU hours. Parallel runs bring that under 1 hour.
- **Main run:** regime-B N = 20 models.
- **Stress test:** regime-A models, which have never seen upright states. This is the clearest "structure helps or doesn't" result.

## Phase 6 — Pre-evaluation demo

Four slides:
1. The problem and the analytical validation residuals.
2. DeLaN structure, with learned M and V against the truth.
3. Rollout and energy-drift plots.
4. MPPI swing-up live with the analytical model and DeLaN.

Freeze the demo commands in the README. The live demo is 20 % of the grade, so it must run from one command.

## Phase 7 — After the demo: ablations, report, submission

**Ablations, kept only if time allows** (the core story does not need them):
- (a) DeLaN with x as an input to M and V, to show what the translational-symmetry prior buys.
- (b) λ_inv ∈ {0, 0.1, 1}.
- (c) The measurement-noise run.
- (d) Closed-loop data efficiency at N = 5.

**Videos:**
- `render.py` writes side-by-side MP4s: analytical | MLP | DeLaN on the same seed, and model rollouts overlaid on MuJoCo.
- The user records or curates the final demo video.

**Report (PDF):** Introduction / Problem statement / Methodology / Results / Conclusion. It maps onto the instruction items as follows:
- Items 1–2 → Phase 0 §0 and Phase 1.
- Item 3 → Phase 2.
- Item 4 → Phase 3.
- Item 5 → Phase 4.
- Item 6 → Phase 5.
- Item 7, the discussion, covers:
  - where the prior helps (energy, velocity-OOR, low data);
  - where it doesn't (angle extrapolation of M and V; the upright instability that punishes every model);
  - compute cost, since the DeLaN JVP plus solve is slower per evaluation than the MLP;
  - the links to A2 (EL derivation, URDF-vs-MuJoCo parameter gaps, integrator-order error) and A3 (model-based control).

Every figure is generated by a script listed in the README, with readable labels and legends.

**Submission zip `Robotics_project_group#no.zip`:** report PDF, code, `requirements.txt`, README, `weights/`, `minutes/`, video (or a link).

**Viva prep (50 %, individual):** every member must be able to derive §0 on a whiteboard:
- M, C and g;
- the skew-symmetry and energy-balance argument;
- why ε, softplus and Cholesky are used;
- how MPPI weights are derived;
- why the inverse loss alone is degenerate.

The `docs/PART*.md` notes double as the viva study guide.

---

## Questions for the instructor meeting (book it before the meeting-request cutoff)

1. Is it acceptable for DeLaN's M and V to take θ only (a translational-symmetry prior), while the MLP gets the full (x, sinθ, cosθ, ẋ, θ̇, τ)?
2. Is it acceptable to include `qfrc_constraint` in the training forces, or should limit rows be dropped?
3. Is the OOR definition (cosθ > 0, plus a velocity-OOR set) acceptable?
4. Is the success definition (|θ| < 0.2 rad held for 2 s, no limit hit) acceptable?

## Risks and mitigations

| risk | mitigation |
|---|---|
| MPPI tuning takes long | Start on the analytical model right after Phase 1. Use the ESS-adaptive λ, the energy-shaping terminal cost, and the MPPI + LQR fallback. |
| DeLaN training is unstable (large M⁻¹) | ε floor, scale-matched initialisation, gradient clipping, a float64 retry. |
| Cart hits the limit in data | Use τ_gen including the constraint force; exclude affected rollout windows. |
| The MLP diverges in rollouts | This is a result, not a bug. First rule out a normalisation error using the training-set one-step error. |
| CPU time for closed loop | Batched torch, Δt_plan = 0.02, multiprocessing over seeds. |
| dm_control install issues | Python 3.12 is supported; `virtualenv` avoids sudo; MUJOCO_GL=egl or osmesa. |

## Verification (end-to-end)

- `pytest tests/` passes:
  - analytical model matches mj_fullM, qfrc_bias, qacc and mj_inverse to ≲1e-10;
  - RK4 matches MuJoCo;
  - the three DeLaN structural tests pass.
- `python scripts/p4_eval.py` produces the metric tables and figures. The analytical row gives ≈0 error, which sanity-checks the pipeline.
- `python scripts/p5_mppi.py --model analytical --seeds 0-19` reaches ≈100 % success before any learned model is evaluated. That confirms the controller works, so later differences come from the models.
- Every figure in the report regenerates from a README command on a clean virtualenv.
