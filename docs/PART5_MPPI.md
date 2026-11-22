# Part 5 — MPPI swing-up

The code is [cartpole/mppi.py](../cartpole/mppi.py) and [scripts/p5_mppi.py](../scripts/p5_mppi.py). The numbers are in [RESULTS.md](RESULTS.md) § Phase 5.

## 1. Algorithm

The state is s = (x, θ, ẋ, θ̇) and the control is u ∈ [−1, 1] with τ = [10u, 0]. One planning iteration, given the current state s₀ and the nominal sequence U ∈ ℝᴴ:

1. **Noise.** Sample K = 512 sequences with an AR(1) filter:
$$\varepsilon_{k,t} = a\,\varepsilon_{k,t-1} + \sqrt{1-a^2}\,\sigma\,n_{k,t}, \qquad a = 0.8,\ \sigma = 0.5,\ n \sim \mathcal N(0,1)$$
   The marginal std stays σ, and the correlation time is −Δt/ln a ≈ 0.09 s, which gives smooth forces.
2. **Clip.** U_k = clip(U + ε_k, −1, 1). The *effective* perturbation ε_k ← U_k − U is used from here on, so the update never pushes U outside its bounds.
3. **Roll out.** All K sequences are rolled out in one batched RK4 pass with Δt_plan = 0.02 s, H = 75 (1.5 s, about one pendulum period, 1.609 s).
4. **Cost.**
$$S_k = \sum_{t=0}^{H-1} c(s_{k,t},u_{k,t}) + 10\,c(s_{k,H},0)$$
$$c = 10(1-\cos\theta) + x^2 + (\dot x/2)^2 + (\dot\theta/5)^2 + 0.1u^2 + 10^3\,\mathrm{relu}(|x|-1.6)^2$$
   Every term is O(1) at its acceptable deviation (Bryson-style weights), and the barrier keeps 0.2 m of margin to a rail that no model knows (ISSUES A2).
5. **Weights.** w_k = softmax(−(S_k − min S)/λ), with λ set by bisection on log λ so that the effective sample size ESS = 1/Σw² lies in [0.1K, 0.2K]. ESS increases monotonically with λ, so the bisection is well posed.
6. **Update.** U ← clip(U + Σ_k w_k ε_k, −1, 1). Apply U₀ for Δt_plan, i.e. 2 simulator steps; shift U by one step, repeating the last entry; and repeat at 50 Hz (ISSUES C1).

**Why this weighting.** MPPI is the importance-sampled solution of the KL-regularised optimal control problem. The optimal control distribution is q*(U) ∝ p(U) exp(−S(U)/λ), with p the sampling distribution, and Σ w_k ε_k is the self-normalised estimate of its mean shift. The information-theoretic version adds λγ Σ_t u_tᵀσ⁻²ε_t to S. With γ = 1 that term spans about ±40 nats here and overwhelmed the path cost (ISSUES C2), so γ = 0. The effort penalty lives in 0.1u² instead.

**Why a statistical temperature.** A fixed λ either collapses onto one sample, when the cost spread is large (ESS → 1), or averages everything, when it is small. Targeting ESS ∈ [0.1K, 0.2K] keeps the estimator's variance and bias in the same regime throughout swing-up and balance, without hand-tuning.

## 2. LQR hand-over (implemented, not used)

`MPPIConfig.lqr = True` switches to u = −K s when |θ| < 0.3 rad and |θ̇| < 2 rad/s, and back when |θ| > 0.5 or |θ̇| > 4 (hysteresis). K comes from continuous-time LQR on the **model's own** Jacobian at upright, by central differences of `model.qdd`, with the pool-S Bryson weights. For the analytical model it reproduces the pool-S gain to 10⁻⁸ (test). Plain MPPI already passed tuning 10/10, so the frozen configuration has `lqr = False`: the model under test drives every control step.

## 3. Making the models fast enough

One plan costs H × 4 = 300 batched model calls (K = 512).

| model in the planner | cost per call | how |
|---|---|---|
| analytical | ~28 µs | closed form, float32 |
| DeLaN | ~110 µs | **lookup table in θ**, float32 |
| MLP | ~1000 µs | float32, standardisation folded into the first and last layers, in-place softplus max(z,0) + log1p(e^{−\|z\|}) |

**Why DeLaN can be tabulated.** DeLaN's M(θ), ∂M/∂θ and ∂V/∂θ depend on θ only (the translational-symmetry prior), so the trained model is 7 scalar periodic functions of θ plus two damping constants. They are sampled once on n = 8192 points and interpolated linearly. The interpolation error is ≤ h²/8 · max|f''| with h = 2π/n. Measured on all test rows it is 2×10⁻⁷ of the q̈ std, about 1000× below DeLaN's own one-step error (test `test_theta_table_matches_delan`). q̇ and τ still enter through the exact Euler–Lagrange algebra. The MLP takes 6 inputs and cannot be tabulated. This is a practical benefit of the structured model: a 34 k-parameter network becomes O(1) table lookups.

## 4. Protocol

- **Plant.** The float64 analytical model + RK4 at h = 0.01 s. It equals MuJoCo to 10⁻¹⁴ off the rail (ISSUES E2). |x| ≥ 1.8 m ends the episode as a rail failure.
- **Initial states.** `dmc_initial_state(seed)` replicates dm_control's swing-up reset exactly: x = 0.01·n, θ = π + 0.01·n, q̇ = 0.01·n from `RandomState(seed)`.
- **Tuning** (no test-seed leakage). Analytical model, seeds 100–109. Configurations are tried in a fixed order, the plan default first, and the first with 10/10 success is frozen. The plan default passed, so no other configuration was tried (`results/p5_tuning.csv`).
- **Evaluation.** Env seeds 0–19. Models: analytical, thin-rod, MLP-B20 and DeLaN-B20 (the seed with the best validation loss), plus a stress test with MLP-A20 and DeLaN-A20, which never saw upright.
- **Metrics:**
  - **balanced:** |wrap θ| < 0.2 and |θ̇| < 1;
  - **time-to-balance:** the first t from which "balanced" holds for 2 s;
  - **success:** balanced within the 10 s episode, and the rail never reached;
  - also: max |x|, the dm_control swing-up reward re-implemented (upright × centred × small control × small velocity, summed over 1000 steps), and the wall time per plan.
- **Statistics.** Wilson 95 % interval for each success rate:
$$\frac{\hat p + z^2/2n \pm z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1+z^2/n}$$
  For example, 20/20 gives [0.84, 1.00] and 18/20 gives [0.70, 0.97]. Only differences the intervals support are claimed.
