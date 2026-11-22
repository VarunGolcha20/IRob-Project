# Part 4 — Open-loop evaluation: definitions

This note defines every Phase 4 metric precisely and says why it is defined that way. The numbers are in [RESULTS.md](RESULTS.md) § Phase 4. The code is [scripts/p4_eval.py](../scripts/p4_eval.py).

## 1. Common setting

- **Models.** Each of the 34 grid models (MLP and DeLaN with λ_inv = 1; regimes A1, A5, A20, B5, B20; all seeds), plus three analytical references:
  - the exact model (the oracle);
  - the thin-rod variant (J 3.2 % low);
  - every parameter scaled by 1.01.
- **Arithmetic.** Every model runs in float64 through the same RK4 with h = 0.01 s and zero-order hold, as MuJoCo does. The backend is [cartpole/npdyn.py](../cartpole/npdyn.py), a numpy port that reproduces the torch models exactly (≤ 1.7×10⁻⁴ relative on the stored validation losses, which is the float32 training gap).
- **Ground truth.** The recorded MuJoCo test pools: R_test (random), E_test (energy pumping), S_test (LQR balance), and U (unforced, undamped). Rail-limit rows are excluded everywhere (ISSUES D2, A2).

## 2. One-step error

For test row i, with label q̈ᵢ (MuJoCo's `qacc`) and prediction q̈̂ᵢ = f(qᵢ, q̇ᵢ, τᵢ):

$$\mathrm{nRMSE}_j(\mathcal R) = \frac{1}{\sigma_j}\sqrt{\frac{1}{|\mathcal R|}\sum_{i\in\mathcal R}(\hat{\ddot q}_{ij}-\ddot q_{ij})^2},\qquad j \in \{\ddot x,\ddot\theta\}$$

- **Normaliser.** σ_j is the std of axis j over *all* test rows (4.80 m/s², 9.66 rad/s²). It is one constant for every region, so regions can be compared with each other. A per-region std would make a region with small accelerations look worse.
- **Combined value.** Tables report √((nRMSE_ẍ² + nRMSE_θ̈²)/2).
- **Regions** (θ = 0 upright):

| region | rule | why |
|---|---|---|
| hanging / mid / upright | cos θ < −0.8 / between / > 0.8 | Hanging dominates the data (ISSUES D4), so an overall mean would hide the upright error. |
| balance | S_test rows | The closed-loop operating region. |
| OOR-angle | E_test rows with cos θ > 0 | Angles that regime A never saw (PART2 §6). |
| OOR-velocity | E_test rows with cos θ < −0.8 and \|θ̇\| > 5.29 rad/s | Familiar angles at unseen speeds. |
| in / out of support | the row's (θ, θ̇) bin holds ≥ 5 rows of *that model's* training set | Support measured from the data each model actually saw. |

- **State error after one step.** One RK4 step from row t, compared with the recorded state at t + 1. Both rows must be limit-free.

## 3. Rollouts and valid-prediction time

- **Windows.** A window starts every 25 steps and lasts 200 steps (2 s). Windows that touch a limit row are dropped. That leaves 214 windows on R_test, 321 on E_test and 165 on S_test.
- **Replay.** The recorded actuator force is replayed open loop. Every model therefore sees exactly the plant's inputs.
- **Error at horizon T** ∈ {0.5, 1, 2} s, for each window:
$$e_\theta(T) = \sqrt{\tfrac{1}{n_T}\textstyle\sum_{k=1}^{n_T}\mathrm{wrap}(\hat\theta_k-\theta_k)^2}$$
  with the analogous expression for x. Wrapping avoids spurious 2π errors after a full rotation.
- **Aggregation.** The median over windows is taken first, then the median over seeds. The median is used because near upright errors grow like e^{λt} with λ = 3.906 /s, so a few windows dominate any mean.
- **Valid-prediction time.** The first t with |wrap(Δθ)| > 0.1 rad, capped at 2 s, averaged over windows. Unlike RMSE, it degrades gracefully after a model diverges. On R_test it saturates at the cap (ISSUES E4).

## 4. Energy drift

The U pool has 40 initial states in 4 groups (PART2 §5). Each is integrated for 1000 steps with τ = 0. The score uses the **true** energy evaluated along the **predicted** trajectory:

$$\Delta H = \max_{0\le t\le 10\,\mathrm s}\frac{|H_{true}(\hat s_t)-H_{true}(s_0)|}{m_p g l},\qquad H_{true} = \tfrac12\dot q^\top M(\theta)\dot q + m_p g l\cos\theta$$

- **Why the true energy.** H_true is the only energy score that is fair to the MLP, which has no energy of its own.
- **DeLaN settings.** DeLaN is run with D = 0 (main setting) and with its learned D (in brackets). The learned D is not identified (ISSUES E3). For the true system, damping changes H by at most 1.1×10⁻⁴ (small), 9.5×10⁻⁴ (large), 1.5×10⁻³ (rotating) and 4.2×10⁻³ (fast_rotating) m_p g l in 10 s (simulated from the same U initial states). That is the same order as the best DeLaN drifts, so a damped comparison would blur exactly the differences being measured. The U ground truth was generated with MuJoCo's damping set to 0, so D = 0 is the like-for-like setting (ISSUES P2).
- **Further outputs:**
  - per group, the median and maximum over its 10 states;
  - DeLaN's drift of its *own* H, which by the identity q̇ᵀ(Ṁ − 2C)q̇ = 0 should sit at the RK4 floor;
  - the drift of p_x;
  - the wrapped-θ error at 10 s against MuJoCo.
- **Floor.** The analytical model reproduces MuJoCo's drift exactly (5.1×10⁻⁹ … 1.0×10⁻⁶), so any value above that comes from the model.

**Why DeLaN's true-energy drift is not zero.** DeLaN conserves its own H_θ exactly, up to the RK4 floor. Along its trajectory H_true(ŝ_t) − H_true(s₀) = [H_true − H_θ](ŝ_t) − [H_true − H_θ](s₀). This is *bounded* by the model error |H_true − H_θ| over the visited states, and it does not accumulate. The MLP has no conserved quantity, so its drift can grow without bound. Hence the regime-A MLP drifts of 5–250 m_p g l on the large and rotating groups.

## 5. Extrapolation

- **Heat map.** The one-step nRMSE is averaged per (θ, θ̇) cell over all test rows, with the cyan contour marking the model's density support.
- **Distance to support.** For each test row, the Chebyshev distance in bins (θ periodic) to the nearest supported bin, using the same 0.1 rad × 0.25 rad/s bins as PART2. The error is reported against this distance. Distances of 20 or more bins are pooled.

## 6. Inverse dynamics and compute

- **Inverse dynamics.** DeLaN gives τ̂ = M q̈ + C q̇ + g + D q̇ for free. On test rows, τ̂_x is compared with the recorded τ_x, and τ̂_θ should be 0 because the hinge is unactuated.
- **Compute.** Mean wall time of one q̈ call on a batch of 512 in numpy float64. Inside MPPI, DeLaN is tabulated in θ (PART5 §3), which removes most of its cost.
