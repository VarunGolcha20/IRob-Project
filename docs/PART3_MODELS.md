# Part 3 — Models and training: MLP baseline and DeLaN

- **Code:** [cartpole/dataset.py](../cartpole/dataset.py), [cartpole/models/mlp.py](../cartpole/models/mlp.py), [cartpole/models/delan.py](../cartpole/models/delan.py), [cartpole/train.py](../cartpole/train.py)
- **Run:**
  - `python scripts/p3_train.py` trains the λ_inv selection and the 34-run grid. It is resumable.
  - `python scripts/p3_structure.py` runs the parameter identification and writes the two figures.
- **Outputs:**
  - `weights/*.pt`, `results/p3_select.json`, `results/p3_training.json`;
  - `results/p3_identification.json`, `figures/p3_learned_structure.*`, `figures/p3_training_curves.*` (written by `p3_structure.py`).
- **Tests:** `pytest -q tests/test_models.py`.
- **Numbers in this note** come from `docs/RESULTS.md` § Phase 3, which comes from `results/p3_training.json` and `results/p3_select.json`. The identification numbers (§8) are not in this note yet, because `p3_structure.py` has not been run.

## 0. Notation

Same as PART1. $q = (x, \theta)$, with $\theta = 0$ upright. $\tau = [10u, 0]^\top$. $M(q)$ is the mass matrix, $C\dot q$ the Coriolis/centripetal force, $g = \partial V/\partial q$ the gravity force, and $D = \mathrm{diag}(d_x, d_\theta) \succeq 0$ the damping. A prime denotes $d/d\theta$.

## 1. What is learned

Both models learn the forward dynamics $\ddot q = f(q, \dot q, \tau_x)$ from rows $(q, \dot q, \tau, \ddot q)$ with exact MuJoCo accelerations (PART2 §1).

| | MLP (black box) | DeLaN |
|---|---|---|
| learns | $f$ directly | $M(\theta)$, $V(\theta)$, $D$; everything else follows from the Euler–Lagrange equations |
| prior | none | Lagrangian structure, $M \succ 0$, translational symmetry |
| parameters | 34 178 | 34 310 |

## 2. DeLaN: from the Lagrangian to $\ddot q$

### 2.1 Euler–Lagrange equations

With $\mathcal L = \tfrac12\dot q^\top M(q)\dot q - V(q)$ and a non-conservative generalized force $\tau - D\dot q$:

$$\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot q} - \frac{\partial\mathcal L}{\partial q} = \tau - D\dot q .$$

Here $\partial\mathcal L/\partial\dot q = M\dot q$, so $\frac{d}{dt}(M\dot q) = M\ddot q + \dot M\dot q$. Also $\partial\mathcal L/\partial q = \tfrac12\partial_q(\dot q^\top M\dot q) - g$. Collecting terms:

$$M\ddot q + \underbrace{\dot M\dot q - \tfrac12\partial_q(\dot q^\top M\dot q)}_{C\dot q} + g + D\dot q = \tau .$$

This defines $C\dot q$ from $M$ alone. No separate Coriolis network is needed, and no Christoffel symbols have to be coded.

### 2.2 Specialising to $M(\theta)$, $V(\theta)$

PART1 P2 shows that $M$ and $V$ depend on $\theta$ only (translational symmetry). Then:
- $\dot M = M'\dot\theta$;
- $\partial_x(\dot q^\top M\dot q) = 0$, and $\partial_\theta(\dot q^\top M\dot q) = \dot q^\top M'\dot q$;
- $g = [0,\ V']^\top$.

With $M'\dot q = [M_{11}'\dot x + M_{12}'\dot\theta,\ M_{12}'\dot x + M_{22}'\dot\theta]^\top$, this gives

$$C\dot q = \dot\theta\, M'\dot q - \tfrac12\begin{bmatrix}0\\ \dot q^\top M'\dot q\end{bmatrix}
= \begin{bmatrix}\dot\theta\,(M_{11}'\dot x + M_{12}'\dot\theta)\\[2pt] \tfrac12 M_{22}'\dot\theta^2 - \tfrac12 M_{11}'\dot x^2\end{bmatrix}.$$

This is what `lagrangian_dynamics` computes. **Check against the truth:** the true $M_{11}$ and $M_{22}$ are constant and $M_{12} = m_p l\cos\theta$, so $C\dot q = [-m_p l\sin\theta\,\dot\theta^2,\ 0]^\top$. That is the standard cart-pole centrifugal term (test 3 verifies it numerically).

The forward dynamics is then

$$\ddot q = M^{-1}\big(\tau - C\dot q - g - D\dot q\big),$$

solved with the closed-form 2×2 `solve2`.

### 2.3 Positive-definite $M$ by construction

$$L = \begin{bmatrix} l_{11} & 0\\ l_{21} & l_{22}\end{bmatrix},\quad l_{11} = \mathrm{softplus}(r_{11}),\ l_{22} = \mathrm{softplus}(r_{22}),\quad M = LL^\top + \epsilon I,$$

$$M_{11} = l_{11}^2 + \epsilon,\qquad M_{12} = l_{11}l_{21},\qquad M_{22} = l_{21}^2 + l_{22}^2 + \epsilon .$$

- $LL^\top \succeq 0$, so every eigenvalue of $M$ is at least $\epsilon$ and $M$ is always invertible. $\epsilon = 10^{-3}$ is 30× below the true $\lambda_{\min} = 0.032$ and bounds $\lVert M^{-1}\rVert \le 1000$, which protects early training.
- The positive diagonal makes the Cholesky factor unique and smooth. With a free-sign diagonal, $l_{11}$ and $-l_{11}$ would give the same $M$.
- Derivatives follow by the product rule: $M_{11}' = 2l_{11}l_{11}'$, $M_{12}' = l_{11}'l_{21} + l_{11}l_{21}'$, $M_{22}' = 2l_{21}l_{21}' + 2l_{22}l_{22}'$, with $l' = \sigma(r)\,r'$ for the softplus entries.

### 2.4 Derivatives by forward-mode propagation

The networks take the scalar $\theta$ through $(\sin\theta, \cos\theta)$ and output $(r_{11}, l_{21}, r_{22})$ and $V$. Their $\theta$-derivatives are needed for $C\dot q$ and $g$, and training backpropagates through them. Each layer therefore carries the pair $(h, h')$, starting from $(\sin\theta, \cos\theta)' = (\cos\theta, -\sin\theta)$:

$$z = Wh + b,\quad z' = Wh',\qquad a = \mathrm{softplus}(z),\quad a' = \sigma(z)\odot z' .$$

The output layer yields $\partial M/\partial\theta$ and $V'$ directly.

- **Cost:** one extra matrix product per layer, the same as one Jacobian–vector product.
- **Why not `torch.func.jvp`:** this is ordinary differentiable torch code. It works under `torch.no_grad()`, which MPPI needs, so no jvp-under-no_grad risk remains (test 6 compares it with autograd to $10^{-10}$ and checks `no_grad` equality).
- **Smoothness:** the network must be at least $C^2$, because the loss differentiates through $M'$ and $V'$. With ReLU, $C\dot q$ and $g$ would be piecewise constant, so every activation is softplus.

### 2.5 The energy identity holds for any $M$ and $V$

Define $H = \tfrac12\dot q^\top M\dot q + V$. Then

$$\dot H = \dot q^\top M\ddot q + \tfrac12\dot q^\top\dot M\dot q + g^\top\dot q .$$

Substituting $M\ddot q = \tau - C\dot q - g - D\dot q$:

$$\dot H = \dot q^\top(\tau - D\dot q) + \dot q^\top\big(\tfrac12\dot M - C\big)\dot q .$$

For $C\dot q = \dot M\dot q - \tfrac12\partial_q(\dot q^\top M\dot q)$, and noting that $\dot q^\top\partial_q(\dot q^\top M\dot q) = \dot q^\top\dot M\dot q$ (a derivative along the path with $\dot q$ held fixed):

$$\dot q^\top C\dot q = \dot q^\top\dot M\dot q - \tfrac12\dot q^\top\dot M\dot q = \tfrac12\dot q^\top\dot M\dot q
\ \Longrightarrow\ \dot q^\top(\dot M - 2C)\dot q = 0 .$$

Hence

$$\boxed{\dot H = \dot q^\top(\tau - D\dot q)}$$

for **any** $M(q)$ and $V(q)$, trained or not. This is a statement about the structure, not about learning quality. With $\tau = 0$ and $D = 0$ a DeLaN conserves *its own* $H$, so any energy drift measured in Phase 4 against the *true* $H$ comes from error in the learned $M$ and $V$, not from the dynamics form. Test 2 checks this on a random-init network: $\dot q^\top(\dot M - 2C)\dot q = 0$ to $10^{-12}$ relative, and $H$ along a 10 s RK4 rollout stays constant to $10^{-6}$ of the kinetic energy.

### 2.6 Damping and its initialisation

$D = \mathrm{diag}(\mathrm{softplus}(d_{\rm raw}))$ keeps $D \succeq 0$, hence passivity. The raw parameter starts at $\mathrm{softplus}^{-1}(10^{-4})$ (**ISSUES P1**). With the default $d_{\rm raw} = 0$, $\mathrm{softplus}(0) = 0.69$ would give a damping time constant of 1.6 s against the true 2200 s, and over 10 s of data the damping barely constrains $D$, so the wrong value would not be corrected. `DeLaN.damping_on = False` zeroes $D$ for the undamped drift test (ISSUES P2).

### 2.7 Inverse dynamics

$\hat\tau = M\ddot q + C\dot q + g + D\dot q$ is available at no extra cost (`DeLaN.inverse`). Test 4 checks $\hat\tau(q, \dot q, \ddot q(\tau)) = \tau$ to $10^{-9}$.

## 3. MLP baseline

The input is $(x, \sin\theta, \cos\theta, \dot x, \dot\theta, \tau_x)$, standardised with training-set mean and std. $\tau_\theta$ is always 0, so feeding it would be a constant input. The body is 3 × 128 softplus, and the output is de-standardised to $(\ddot x, \ddot\theta)$. It uses the same softplus as DeLaN, so smoothness is not a confound.

**Parameter counts:**

| | layers | count |
|---|---|---|
| MLP | $6\!\to\!128\!\to\!128\!\to\!128\!\to\!2$ | $896 + 16\,512 + 16\,512 + 258 = 34\,178$ |
| DeLaN L-net | $2\!\to\!128\!\to\!128\!\to\!3$ | $384 + 16\,512 + 387 = 17\,283$ |
| DeLaN V-net | $2\!\to\!128\!\to\!128\!\to\!1$ | $384 + 16\,512 + 129 = 17\,025$ |
| DeLaN damping | | 2 |
| DeLaN total | | 34 310 |

## 4. Data and normalisation (`cartpole/dataset.py`)

- **Regime A:** $N$ episodes of R_train. **Regime B:** $N$ of R, $N$ of E and $\lceil N/2\rceil$ of S. Validation is R_val (A) or R_val + E_val + S_val (B).
- Rail-limit rows are always dropped (ISSUES D2).
- The episode subset comes from `np.random.default_rng([seed, N, regime])`, so each seed gets its own subset (ISSUES D6) and the draw depends on nothing else.
- **Statistics** (training rows only): per-feature input mean and std (MLP), per-axis std $\sigma_{\ddot x}, \sigma_{\ddot\theta}$ (loss), and $\sigma_\tau$ (inverse loss).
- Training runs in float32. Phase 4 evaluations run in float64.

## 5. Loss

**Forward loss** (identical for both models):

$$\mathcal L_{fwd} = \frac1n\sum_i\Big[\Big(\frac{\hat{\ddot x}_i - \ddot x_i}{\sigma_{\ddot x}}\Big)^2 + \Big(\frac{\hat{\ddot\theta}_i - \ddot\theta_i}{\sigma_{\ddot\theta}}\Big)^2\Big].$$

Per-axis normalisation is needed because $\sigma_{\ddot\theta}$ is 2–3× $\sigma_{\ddot x}$ (PART2 §2); without it, the $\theta$ axis would dominate.

**Inverse loss** (DeLaN only, weight $\lambda_{inv}$):

$$\mathcal L_{inv} = \Big(\frac{\hat\tau_x - \tau_x}{\sigma_\tau}\Big)^2 + \Big(\frac{\hat\tau_\theta}{\mathrm{sg}(M_{22})\,\sigma_{\ddot\theta}}\Big)^2 .$$

The second term has target 0, because the pole is unactuated. **Why the normaliser matters.** $\hat\tau_\theta = M_{21}\ddot x + M_{22}\ddot\theta + (C\dot q)_2 + V' + d_\theta\dot\theta$ is built from the learned $\theta$-row quantities. If its residual were divided by a *fixed* scale, the network could shrink it by scaling the whole row down ($M_{22}$, $V'$, $d_\theta$ together), a degenerate direction that satisfies "target 0" without fitting the physics. Dividing by the model's own detached $M_{22}$ turns the residual into $\hat\tau_\theta/M_{22} = \ddot\theta_{\rm implied} - \ddot\theta$, an *acceleration* error, which scaling the row does not change. The stop-gradient (sg) keeps $M_{22}$ from being driven down to shrink the denominator. The forward loss always stays in.

**Selecting $\lambda_{inv}$ ∈ {0, 0.1, 1}** (rule fixed in advance in PLAN 3.3): lowest mean validation loss over A20 and B20, seed 0. $\lambda_{inv} = 1$ won:

| | λ = 0 | λ = 0.1 | λ = 1 |
|---|---|---|---|
| A20 | 1.04×10⁻⁷ | 9.1×10⁻⁸ | 7.9×10⁻⁸ |
| B20 | 1.94×10⁻⁷ | 2.23×10⁻⁷ | 1.84×10⁻⁷ |
| mean | 1.49×10⁻⁷ | 1.57×10⁻⁷ | **1.31×10⁻⁷** |

This is a 12 % margin on a single seed, and the ordering is not monotone (λ = 0.1 is worse than λ = 0 on B20), so it is **not shown to be significant** (ISSUES T2). The inverse loss also doubles DeLaN's cost per step. The Phase 7 ablation (λ = 0 at seeds 1–2) will measure real seed spread. *Source: `results/p3_select.json`.*

## 6. Training procedure (`cartpole/train.py`)

| setting | value | reason |
|---|---|---|
| optimiser | Adam, lr $10^{-3}$, cosine decay to $10^{-5}$ | |
| batch | 512 | |
| gradient clipping | norm 10 | protection against early $M^{-1}$ spikes (ISSUES T1) |
| budget | 30 000 steps for every run | equal compute across $N$; epochs are not comparable between $N = 1$ (about 1000 rows) and $N = 20$ |
| validation | every 500 steps; best checkpoint kept | |
| early stopping | patience 10 evaluations (5000 steps) | **under review, ISSUES T3** |
| precision | float32 | |
| seeds | 5 for $N = 1$, 3 for $N = 5, 20$; each seed changes the subset and the initialisation | ISSUES D6 |
| runs | A: (5 + 3 + 3) × 2 = 22; B: (3 + 3) × 2 = 12; total 34 | |

**Initialisation without the true parameters:** PyTorch defaults for the weights, and the L-diagonal biases at 0, which gives $M \approx 0.48\,I$ (the data-driven $M_{11}$ initialisation from PLAN 3.2 is not implemented; it stays a fallback). The damping starts at $10^{-4}$ (§2.6).

**Cost per gradient step** (batch 512): DeLaN costs about 2.6× the MLP, and the inverse loss doubles DeLaN's cost again.

## 7. Results and what they do and do not show

Median best validation loss over seeds, with [min, max]. *Source: `results/p3_training.json`; full per-seed table in RESULTS.md.*

| config | MLP | DeLaN | MLP / DeLaN |
|---|---|---|---|
| A1 (5 seeds) | 2.7×10⁻³ [8.2×10⁻⁴, 3.6×10⁻²] | 6.4×10⁻⁵ [3.1×10⁻⁵, 6.0×10⁻⁴] | 43× |
| A5 (3 seeds) | 1.1×10⁻⁵ [8.0×10⁻⁶, 1.9×10⁻⁵] | 1.8×10⁻⁷ [4.8×10⁻⁸, 3.8×10⁻⁶] | 60× |
| A20 (3 seeds) | 7.1×10⁻⁶ [6.6×10⁻⁶, 8.9×10⁻⁶] | 4.5×10⁻⁷ [7.9×10⁻⁸, 1.2×10⁻⁶] | 16× |
| B5 (3 seeds) | 6.2×10⁻⁵ [2.8×10⁻⁵, 8.0×10⁻⁵] | 2.3×10⁻⁶ [1.7×10⁻⁷, 9.8×10⁻⁶] | 27× |
| B20 (3 seeds) | 2.5×10⁻⁵ [2.2×10⁻⁵, 2.6×10⁻⁵] | 1.8×10⁻⁷ [1.8×10⁻⁷, 2.9×10⁻⁶] | 140× |

- DeLaN's median validation loss is 16× to 140× below the MLP's in every configuration.
- DeLaN varies much more between seeds: up to 79× (A5), against at most 2.9× for the MLP outside A1.
- Most $N \ge 5$ runs reached their best checkpoint at 29 000–30 000 steps. The step budget, not convergence, therefore sets their final accuracy.
- These are validation losses only. They say nothing about rollouts, energy, or extrapolation, which are Phase 4 questions.

**T3 (open): early stopping cut some DeLaN runs short.** Ten runs stopped before 30 000 steps. Three are DeLaN at $N \ge 5$ (B5 seeds 1 and 2, B20 seed 1, at 14 500–15 000 steps), five are DeLaN A1, and two are MLP A1. The three DeLaN $N \ge 5$ runs ended in a stretch of loss spikes while the learning rate was still near $0.75\times10^{-3}$, and their best losses are 10–60× above the DeLaN seeds that ran to the end. The bias is *against* DeLaN, so the present comparison is conservative, but the seed spread and the "best seed" choice are affected. The proposed fix is to drop early stopping for every run, train the full 30 000 steps and keep the best-validation checkpoint. Only the ten runs above would change. **Numbers in this section describe the grid as currently trained and must be refreshed if T3 is fixed.**

## 8. Parameter identification (`scripts/p3_structure.py`)

The learned $M$ and $V$ are compared with the physics by least squares on the *physical* basis, evaluated at the training angles of each run (so only states the model was trained on count):

| learned quantity | fitted basis | gives | true value |
|---|---|---|---|
| $M_{11}(\theta)$ | constant (mean) | $m_c + m_p$ | 1.1 kg |
| $M_{12}(\theta)$ | $\alpha\cos\theta$ | $m_p l$ (**inertial**) | 0.05 kg m |
| $M_{22}(\theta)$ | constant (mean) | $J$ | 0.0344246 kg m² |
| $V'(\theta)$ | $\beta\,(-g\sin\theta)$ | $m_p l$ (**gravitational**) | 0.05 kg m |

- **Two independent estimates of $m_p l$.** One comes from the inertia matrix and one from gravity. They are learned by different parts of the network, so their agreement is a physics check that the MLP cannot offer.
- **Fit on $V'$, not on $V$.** $V$ is defined only up to an additive constant. The gravity torque $V'$ is not, so the fit needs no free constant. (PLAN 3.6 wrote "fit $V$ on $\cos\theta$"; this is the equivalent, better-posed form.)
- **No free constant in the $M_{12}$ fit.** In regime A, $\cos\theta \approx -1$ throughout, so $\cos\theta$ and a constant are almost collinear and the split between them would be arbitrary.
- The script also reports the fit residuals, the consistency $\alpha/\beta - 1$, the learned damping, and the errors over the full circle $[-\pi, \pi]$ (meaningful only where trained).
- **Expectation, to check rather than assume:** regime A sees only angles near hanging, so $M$ and $V$ are identified well there and not elsewhere (ISSUES T1: a narrow $\theta$ range makes $M$ weakly identifiable). Regime B covers the upright region and should identify more of the circle.

**Results:** *pending. Run `python scripts/p3_structure.py`, then record `results/p3_identification.json` in RESULTS.md and add the table here.*

## 9. Tests (`tests/test_models.py`)

| test | what it checks |
|---|---|
| 0 | parameter counts matched: MLP 34 178, DeLaN 34 310 |
| 1 | random-init $M$ symmetric with eigenvalues $\ge\epsilon$ over $10^4$ values of $\theta$ |
| 2 | $\dot q^\top(\dot M - 2C)\dot q = 0$ to $10^{-12}$; own-$H$ conserved over a 10 s RK4 rollout (§2.5) |
| 3 | analytical $M$, $V$ in `lagrangian_dynamics` reproduce `AnalyticalModel.qdd` to $10^{-11}$ (checks the Coriolis code) |
| 4 | inverse ∘ forward is the identity to $10^{-9}$ |
| 5 | both models fit 256 real rows: loss drops over 1000× and ends below $10^{-2}$ (a plumbing check; ISSUES T1) |
| 6 | forward-mode $\theta$-derivatives match autograd to $10^{-10}$; `qdd` is identical under `torch.no_grad()` |

## 10. Status against the acceptance criteria (PLAN 3.6)

| criterion | status |
|---|---|
| tests 0–6 pass | passing (suite 40/40 at the last run recorded in PLAN) |
| validation loss converges for every run | **partly**: most $N \ge 5$ runs are still improving at the 30 000-step budget (§7); T3 open |
| A20 in-range one-step nRMSE of a few percent or less | Phase 4 |
| identified parameters within a few percent of truth inside the training support | pending (§8) |

## 11. Limits to state in the report

- Validation loss is not test error. Phase 4 supplies the test-set claims.
- DeLaN's $M$ and $V$ see only $(\sin\theta, \cos\theta)$, a translational-symmetry prior the MLP does not get (instructor question 1 in PLAN). The Phase 7 ablation with $x$ as an input measures what this prior buys.
- $\lambda_{inv} = 1$ rests on a thin single-seed margin (T2).
- Seed spread is large for DeLaN, so claims should use medians and ranges, not single runs.
