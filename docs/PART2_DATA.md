# Part 2 — Data generation

- **Code:** [cartpole/data.py](../cartpole/data.py), [scripts/p2_collect.py](../scripts/p2_collect.py)
- **Run:** `python scripts/p2_collect.py`, which takes about 50 s on CPU.
- **Outputs:**
  - `data/{R,E,S}_{train,val,test}.npz`, `data/U.npz` (15 MB in total);
  - `results/p2_dataset.json`;
  - `figures/p2_coverage.*`.
- **Tests:** `pytest -q tests/test_data.py` audits every label.

## 1. What one row contains

Row $t$ of an episode is recorded **before** the step. We set `ctrl = u_t`, call `mj_forward`, and read out:

$$\big(x,\theta,\dot x,\dot\theta\big)_t,\; u_t,\; \texttt{qfrc\_actuator},\; \texttt{qfrc\_constraint},\; \texttt{qfrc\_passive},\; \texttt{qfrc\_bias},\; \ddot q_t = \texttt{qacc},\; M(q_t),\; E_t,\; \text{limit\_active}.$$

This is the PDF's `physics.forward()` recipe, shifted by one step. The shift aligns each state with the action applied *from* it, so that $s_{t+1} = \mathrm{RK4}(s_t, u_t)$ holds exactly. Rollouts can then replay the recorded actions without an off-by-one error.

- **Labels are exact.** $\ddot q_t$ is MuJoCo's continuous-time acceleration, not a finite difference, so no differentiation noise enters the targets.
- **Generalized force.**
  $$\tau_{gen} = \texttt{qfrc\_actuator} + \texttt{qfrc\_constraint} = \begin{bmatrix}10u + f_{rail} \\ 0\end{bmatrix}.$$
  The rail-limit force is just another generalized force in the Euler–Lagrange equations (PART1 §4, P6), so rows with the limit active are *correct* data.
- **Training drops the limit rows anyway** (`flatten(..., drop_limit=True)`; [ISSUES.md](ISSUES.md) D2). On those rows the constraint force alone reaches 61 N in R_train, 67 N in E_train and 178 N in S_train (a falling episode hitting the rail), against the actuator's ±10 N; the total force τ_x reaches 53 N in R_train. It is impulsive, and it never occurs in any test-time use: rollouts exclude windows that touch the rail, and MPPI keeps $\lvert x\rvert < 1.6$ m. Keeping the rows would give the MLP force inputs it never sees in use and would stretch its input normalisation, while DeLaN, being linear in $\tau$, would be unaffected. That asymmetry has nothing to do with the physics prior we want to test. Dropping the rows costs 4 % of R and under 1 % of E. Both models are trained on the same rows.
- **Label audit** (`tests/test_data.py`). Every row of every split satisfies $M\ddot q + C\dot q + g + D\dot q = \tau_{gen}$ to $<10^{-10}$, or $<10^{-3}$ on rail-limit rows, where the constraint solver's tolerance applies (PART1, check 4b). The actuator force is exactly $[10u, 0]$.

## 2. Pools and splits

Splits are made **by episode**. Consecutive rows at 100 Hz differ by $O(h)$, so a row-level split would put near-copies of test rows into training. Each episode uses one seed, which sets both the env's initial state and the policy noise, and no seed appears in two splits (tested).

| pool | excitation | train | val | test | seeds |
|---|---|---|---|---|---|
| **R** random | $u_t \sim U[-1,1]$ iid per step (statement / PDF) | 20 | 5 | 10 | 0–19 / 100–104 / 200–209 |
| **E** energy-pumping | §4 below | 20 | 5 | 10 | 300–319 / 400–404 / 500–509 |
| **S** balance | LQR about upright + OU noise, §4b below | 10 | 3 | 5 | 700–709 / 800–802 / 900–904 |
| **U** unforced, undamped | $u = 0$, `dof_damping[:] = 0`, 40 chosen initial states | — | — | 40 | 600–639 |

Each episode lasts 10 s, i.e. 1000 rows, the dm_control default. Initial states for R and E come from dm_control's swing-up reset: $\theta = \pi + 0.01\,\mathcal N$, with $x$ and the velocities also $0.01\,\mathcal N$. S starts near upright (§4b).

| split | rows | rows without limit | rail limit active | max $\lvert x\rvert$ | max $\lvert\dot\theta\rvert$ | max $\cos\theta$ | above horizontal | balance region* | std $\ddot x$, $\ddot\theta$ | $u$ saturated |
|---|---|---|---|---|---|---|---|---|---|---|
| R_train | 20 000 | 19 170 | 4.2 % | 1.86 | 5.29 | −0.229 | **0 %** | 0 | 5.8, 9.6 | 0.1 % |
| R_val | 5 000 | 4 758 | 4.8 % | 1.86 | 5.27 | −0.108 | 0 % | 0 | 5.8, 9.8 | 0.1 % |
| R_test | 10 000 | 9 546 | 4.5 % | 1.86 | 4.21 | −0.457 | 0 % | 0 | 5.8, 9.5 | 0.1 % |
| E_train | 20 000 | 19 879 | 0.6 % | 1.89 | 8.40 | 1.000 | 39 % | **128** | 4.1, 11.5 | 7.7 % |
| E_val | 5 000 | 5 000 | 0 % | 1.57 | 7.70 | 0.964 | 29 % | 0 | 3.7, 11.9 | 4.6 % |
| E_test | 10 000 | 9 968 | 0.3 % | 1.84 | 8.45 | 1.000 | 36 % | 41 | 4.0, 11.4 | 7.8 % |
| S_train | 10 000 | 9 786 | 2.1 % | 1.99 | 9.91 | 1.000 | 94 % | **6 797** | 6.9, 8.7 | 22 % |
| S_val | 3 000 | 2 840 | 5.3 % | 2.01 | 9.10 | 1.000 | 94 % | 1 739 | 10.3, 11.0 | 30 % |
| S_test | 5 000 | 5 000 | 0 % | 1.15 | 2.93 | 1.000 | 100 % | 4 241 | 4.8, 6.0 | 10 % |

\*Balance region: $\lvert\mathrm{wrap}(\theta)\rvert < 0.2$ rad and $\lvert\dot\theta\rvert < 1$ rad/s, the region where MPPI has to hold the pole (PLAN Phase 5 success criterion).

**Loss normalisation follows from this table.** The std of $\ddot\theta$ is about 2× that of $\ddot x$ in R and about 3× in E. A raw MSE would therefore mostly fit $\ddot\theta$. Phase 3 normalises each axis by its training std.

## 3. Why random actions never reach upright: a random walk in energy

**Reduced (internal) energy.** Use Routh reduction on the cyclic coordinate $x$, with $p_x = M_t\dot x + a\cos\theta\,\dot\theta$ and $a = m_p l$. The total energy splits as

$$H = \frac{p_x^2}{2M_t} + \underbrace{\tfrac12 J_{eff}(\theta)\dot\theta^2 + a g\cos\theta}_{\text{internal}}, \qquad J_{eff}(\theta) = J - \frac{a^2\cos^2\theta}{M_t}.$$

The internal energy above hanging, $\tilde E = \tfrac12 J_{eff}\dot\theta^2 + a g(1 + \cos\theta)$, must reach $2ag = 0.981$ J for the pole to stand up.

**Linear theory.** Linearise about hanging, $\varphi = \theta - \pi$, using PART1 P4:

$$\ddot\varphi + \omega^2\varphi = \beta F, \qquad \beta = \frac{a}{M_t J_{eff,0}} = 1.414\ \text{(kg m}^2)^{-1},\quad J_{eff,0} = J - a^2/M_t.$$

The force is held for one step $h$ and then redrawn independently, with variance $\sigma_F^2 = 10^2/3$. Each step therefore kicks $\dot\varphi$ by $\beta F h$. The kick is independent of the state, so the cross term has zero mean and

$$\mathbb E[\Delta(\dot\varphi^2)] = \beta^2\sigma_F^2 h^2 \text{ per step}
\;\Rightarrow\;
\mathbb E[\tilde E](t) = \tfrac12 J_{eff,0}\,\beta^2\sigma_F^2 h\; t = 0.0107\ \text{J/s} \times t,$$

using $\tilde E \approx J_{eff,0}\cdot\tfrac12(\dot\varphi^2 + \omega^2\varphi^2)$ and $\omega^2 J_{eff,0} = ag$. This rate depends only on the force variance per unit time, $\sigma_F^2 h$. With 100 Hz iid noise that variance is spread thinly in frequency, so very little of it lands at the 0.62 Hz resonance.

**Consequence.** The *expected* time to accumulate the upright energy is $2ag / 0.0107 \approx$ **92 s**, nine times the episode length. After 10 s the expected energy is only 11 % of what upright needs.

**Measured** (all 35 R episodes; figure panel b):

| | rate [J/s] |
|---|---|
| linear theory | 0.01071 |
| measured, each episode until its first rail contact | 0.01036 (−3 %) |
| measured, whole episodes | 0.0149 |

The theory holds until the cart reaches the rail, which happens at a median of 4.3 s. After that, the rail impacts (constraint forces up to 61 N in R_train, six times the actuator range) add energy faster. Even so, the pole never rises above horizontal in any R episode: the largest $\cos\theta$ is −0.229 in R_train. So the upright region has **zero support** in the random data, as the problem statement anticipated. A second dataset is needed.

## 4. Energy-pumping excitation (pool E)

**Pumping law.** The pole equation $a\cos\theta\,\ddot x + J\ddot\theta - ag\sin\theta = 0$ gives, for the pole's energy in the cart frame $E_p = \tfrac12 J\dot\theta^2 + ag\cos\theta$,

$$\dot E_p = -a\cos\theta\,\dot\theta\,\ddot x.$$

Command the cart acceleration as

$$\ddot x_{des} = k_E\,\frac{E_p - E_{ref}}{ag}\,\dot\theta\cos\theta \;-\; k_x x - k_v\dot x.$$

Ignoring the centering terms,
$$\frac{d}{dt}\tfrac12(E_p - E_{ref})^2 = -\frac{k_E\,a}{ag}(E_p - E_{ref})^2(\dot\theta\cos\theta)^2 \le 0,$$
so $E_p$ converges to $E_{ref}$ (Åström–Furuta). The action is $u = \mathrm{clip}\big(M_t\ddot x_{des}/10 + n_t\big)$.

**Choices and reasons:**

| choice | reason |
|---|---|
| $E_{ref} \sim U[-0.5, 1.5]\,ag$, redrawn every 2.5 s (≈1.5 periods) | Spans small swings, the separatrix ($E_{ref} = ag$, which brings the pole to the upright slowly, giving near-balance states) and full rotations, all within one episode. |
| $k_E = 6$ m/s² | With $\lvert E_p - E_{ref}\rvert \sim ag$ and $\lvert\dot\theta\rvert \sim 4$, the commanded acceleration is about 24 m/s², so pumping is fast. The actuator saturates only 7.7 % of the time. |
| $k_x = k_v = 3$ | The centering loop is $s^2 + 3s + 3$: $\omega_n = 1.7$ rad/s, $\zeta = 0.87$. One test: gains of 1.5 gave 2.3 % rail-limit rows, gains of 3 gave 0.4 %, with the same 43 % above horizontal. |
| OU noise, $\tau_c = 0.2$ s, $\sigma = 0.3$ | Its corner frequency, $1/(2\pi\tau_c) = 0.8$ Hz, is close to the 0.62 Hz resonance, so the noise power goes where it moves the pole. It also breaks the $\dot\theta\cos\theta = 0$ deadlock at rest. 0.3 (3 N) explores without overriding the pumping law. |
| nominal parameters in the policy | The policy only collects data and is never evaluated, so using the true $J$ and $a$ biases no comparison. |

**Result.** In E_train the pole spends 39 % of the time above horizontal, reaches $\cos\theta = 1.000$, and makes up to 3.2 net turns per episode (figure panels a and c).

**Gap.** The pumping law takes the pole *through* upright with speed. Only 128 E_train rows (0.64 %, from 3 of 20 episodes) lie in the balance region, which is exactly where MPPI must hold the pole ([ISSUES.md](ISSUES.md) D3). Pool S fills this gap.

## 4b. Balance excitation (pool S)

The policy is an LQR about the upright, plus OU noise. These are the "controller rollouts" the course instructions list as an allowed excitation.

**Linearisation.** Take PART1 §3 at $q = 0$, with state $s = (x, \theta, \dot x, \dot\theta)$ and $\theta$ wrapped to $(-\pi, \pi]$:
$$M_0\ddot q = \begin{bmatrix}10u \\ 0\end{bmatrix} + \begin{bmatrix}0 \\ ag\,\theta\end{bmatrix} - D\dot q, \qquad M_0 = \begin{bmatrix}M_t & a \\ a & J\end{bmatrix},$$
which gives
$$A = \begin{bmatrix}0 & I \\ M_0^{-1}G & -M_0^{-1}D\end{bmatrix},\quad G = \begin{bmatrix}0 & 0 \\ 0 & ag\end{bmatrix},\quad B = \begin{bmatrix}0 \\ M_0^{-1}[10, 0]^\top\end{bmatrix}.$$

**Weights by Bryson's rule,** $Q_{ii} = 1/s_{i,max}^2$, with the largest acceptable deviations $\lvert x\rvert = 1$ m, $\lvert\theta\rvert = 0.3$ rad, $\lvert\dot x\rvert = 2$ m/s, $\lvert\dot\theta\rvert = 2$ rad/s, and $R = 1$ because $\lvert u\rvert \le 1$. Solving the continuous algebraic Riccati equation gives $K = [-1.00, -8.60, -1.34, -2.14]$ and closed-loop eigenvalues $-7.26 \pm 1.36j$ and $-1.36 \pm 0.83j$, all stable (tested).

**Noise and initial states.** The OU noise has $\tau_c = 0.2$ s, and its amplitude is drawn per episode as $\sigma \sim U[0.5, 1.5]$. Initial states satisfy $\lvert\theta\rvert \le 0.3$, $\lvert\dot\theta\rvert \le 1$, $\lvert x\rvert \le 0.5$, $\lvert\dot x\rvert \le 0.5$.

**Choosing σ, two checks:**

| σ range | balance-region share | 0.2 < \|θ\| < 0.6 | falls |
|---|---|---|---|
| [0.2, 0.6] | ≈ 99 % | ≈ 0 (max \|θ\| = 0.26) | 0 / 8 |
| **[0.5, 1.5]** (used) | 68 % | 13 % | 1 / 10 |

With small noise the LQR holds the pole too well, so the data covers only a tight core. MPPI arrives at upright with speed after the swing, so the model also needs the surrounding shell and some falling states. One S_train episode (σ = 1.42) and one S_val episode lose the pole. That is useful: it adds "falling from upright" states.

**Result.** S_train adds 6 797 balance-region rows, against 128 in E_train. Regime A does not use S, so the extrapolation story for random-only models is unchanged.

**Nominal parameters.** Like the pumping law, the LQR uses the true model. It collects data and is never evaluated ([ISSUES.md](ISSUES.md) D9).

## 5. Unforced, undamped pool (U): ground truth for the energy-drift test

`model.dof_damping[:] = 0` and $u = 0$, so the true energy is conserved. Initial states come in four groups of 10, with the cart velocity chosen so that $p_x = 0$. The system's COM therefore stays fixed, and the cart moves at most 9 cm, so the rail is never reached.

**Group boundaries.** The separatrix lies at internal energy $2ag$. At the bottom it corresponds to $\lvert\dot\theta\rvert = \sqrt{4ag/J_{eff,0}} = 7.81$ rad/s. "Small" and "large" stay below it, while "rotating" and "fast_rotating" lie above it.

**Coverage of each group.** A test is only "in range" if the model was trained there ([ISSUES.md](ISSUES.md) D5), so each group carries the fraction of its rows that fall in each regime's training support. Support is defined by density (§6).

| group | initial state | internal energy / $ag$ | in regime-A support | in regime-B support | MuJoCo max $\lvert\Delta H\rvert/(ag)$, 10 s |
|---|---|---|---|---|---|
| small | $\varphi_0 \in [-0.5, 0.5]$, $\dot\theta_0 \in [-1, 1]$ | 0.01–0.11 | 100 % | 100 % | 5.1×10⁻⁹ |
| large | released from rest, $\cos\theta_0 \in [-0.5, 0.95]$ | 0.52–1.93 | 15 % | 85 % | 8.5×10⁻⁸ |
| rotating | at the bottom, $\lvert\dot\theta_0\rvert \in [7.9, 8.3]$ (just above the separatrix) | 2.08–2.25 | 0 % | 87 % | 2.2×10⁻⁷ |
| fast_rotating | at the bottom, $\lvert\dot\theta_0\rvert \in [8.5, 11]$ | 2.41–3.87 | 0 % | **1 %** | 1.0×10⁻⁶ |

"fast_rotating" lies outside *every* training set. It is kept on purpose as an extrapolation test, and its drift must be reported as extrapolation.

**Integrator floor.** MuJoCo's drift matches our RK4 floor (PART1 §5). Any learned-model drift above about $10^{-6}\,ag$ comes from the model.

## 6. Out-of-range (OOR) test sets

The boundaries come from the training data, not from a guess. They are computed on R_train: $\max\cos\theta = -0.229$ and $\max\lvert\dot\theta\rvert = 5.29$ rad/s. Rows from E_test where the rail limit is active are excluded.

**Caveat: rail impacts set the max-based boundary** ([ISSUES.md](ISSUES.md) D1). 18 of 20 R_train episodes hit the rail. The extremes compare as follows:

| R_train boundary | $\cos\theta$ | $\lvert\dot\theta\rvert$ |
|---|---|---|
| max over all rows (used for the set definitions) | −0.229 | 5.29 |
| 99th percentile | −0.420 | 4.18 |
| max before each episode's first rail contact | −0.559 | 4.07 |

A max-based box therefore overstates the support. We also define a **density-based support**: a state is supported if its $(\theta, \dot\theta)$ bin, of size 0.1 rad × 0.25 rad/s, holds at least 5 training rows (0.05 s of data). The plane is $(\theta, \dot\theta)$ because the dynamics do not depend on $x$ (PART1, P2). This is `cartpole.data.in_support`. With it:
- of the 3 677 E_test rows inside the max-based box, only **1 444** are density-supported;
- **none** of the OOR-angle or OOR-velocity rows are density-supported, so both OOR sets are truly out of range.

Phase 4 reports error against distance from the support, in addition to these sets.

| set | definition | E_test rows | what it tests |
|---|---|---|---|
| OOR-angle | $\cos\theta > 0$ (pole above horizontal) | 3 596 (of which 153 also have $\lvert\dot\theta\rvert > 5.29$) | Extrapolation of $M(\theta)$ and $V(\theta)$ to angles never seen. DeLaN's networks must extrapolate here too; the structure alone does not guarantee it. |
| OOR-velocity | $\cos\theta < -0.8$ and $\lvert\dot\theta\rvert > 5.29$ | 1 046 | Extrapolation in velocity at familiar angles. DeLaN's $\ddot q$ is exactly quadratic in $\dot q$ (Coriolis) and linear in $\tau$, so the structure should carry it. The MLP has no such guarantee. |
| in R support | $\cos\theta \le -0.229$ and $\lvert\dot\theta\rvert \le 5.29$ | 3 677 | Control set: E-pool states inside the R support. |

Separating angle from velocity is deliberate. A single "upright" number would mix extrapolation that the Lagrangian structure guarantees (in $\dot q$ and $\tau$) with extrapolation that only the networks can provide (in $\theta$).

## 7. Training regimes (used in Phase 3)

All regimes train on rows without the rail limit active (§1).

- **A — random only.** $N \in \{1, 5, 20\}$ R_train episodes. For each training seed, the $N$ episodes are a random subset drawn with that seed, so the reported mean ± std covers both data variability and initialisation. Regime A is used for one-step error, rollouts, drift, OOR and data efficiency.
- **B — random + pumping + balance.** $N$ R_train, $N$ E_train and $\lceil N/2\rceil$ S_train episodes: (1, 1, 1), (5, 5, 3) or (20, 20, 10). $N = 20$ is the main model for MPPI, so every model has seen upright states, including the balance region.
- **Validation.** Early stopping uses R_val in regime A, and R_val + E_val + S_val in regime B. Test sets are never touched during training or model selection.
