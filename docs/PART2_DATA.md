# Part 2 — Data generation

- **Code:** [cartpole/data.py](../cartpole/data.py), [scripts/p2_collect.py](../scripts/p2_collect.py)
- **Run:** `python scripts/p2_collect.py`, which takes about 50 s on CPU.
- **Outputs:**
  - `data/{R,E}_{train,val,test}.npz`, `data/U.npz` (12 MB in total);
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
  The rail-limit force is just another generalized force in the Euler–Lagrange equations (PART1 §4, P6). Both the MLP and DeLaN are trained with $\tau_{gen}$, so the rows where the cart hits the rail stay usable.
- **Label audit** (`tests/test_data.py`). Every row of every split satisfies $M\ddot q + C\dot q + g + D\dot q = \tau_{gen}$ to $<10^{-10}$, or $<10^{-3}$ on rail-limit rows, where the constraint solver's tolerance applies (PART1, check 4b). The actuator force is exactly $[10u, 0]$.

## 2. Pools and splits

Splits are made **by episode**. Consecutive rows at 100 Hz differ by $O(h)$, so a row-level split would put near-copies of test rows into training. Each episode uses one seed, which sets both the env's initial state and the policy noise, and no seed appears in two splits (tested).

| pool | excitation | train | val | test | seeds |
|---|---|---|---|---|---|
| **R** random | $u_t \sim U[-1,1]$ iid per step (statement / PDF) | 20 | 5 | 10 | 0–19 / 100–104 / 200–209 |
| **E** energy-pumping | §4 below | 20 | 5 | 10 | 300–319 / 400–404 / 500–509 |
| **U** unforced, undamped | $u = 0$, `dof_damping[:] = 0`, 30 chosen initial states | — | — | 30 | 600–629 |

Each episode lasts 10 s, i.e. 1000 rows, the dm_control default. Initial states for R and E come from dm_control's swing-up reset: $\theta = \pi + 0.01\,\mathcal N$, with $x$ and the velocities also $0.01\,\mathcal N$.

| split | rows | rail limit active | max $\lvert x\rvert$ | max $\lvert\theta - \pi\rvert$ | max $\lvert\dot\theta\rvert$ | max $\cos\theta$ | above horizontal | std $\ddot x$, $\ddot\theta$ | $u$ saturated |
|---|---|---|---|---|---|---|---|---|---|
| R_train | 20 000 | 4.2 % | 1.86 | 1.34 | 5.29 | −0.229 | **0 %** | 5.8, 9.6 | 0.1 % |
| R_val | 5 000 | 4.8 % | 1.86 | 1.46 | 5.27 | −0.108 | 0 % | 5.8, 9.8 | 0.1 % |
| R_test | 10 000 | 4.5 % | 1.86 | 1.10 | 4.21 | −0.457 | 0 % | 5.8, 9.5 | 0.1 % |
| E_train | 20 000 | 0.6 % | 1.89 | π | 8.40 | 1.000 | 39 % | 4.1, 11.5 | 7.7 % |
| E_val | 5 000 | 0 % | 1.57 | 2.87 | 7.70 | 0.964 | 29 % | 3.7, 11.9 | 4.6 % |
| E_test | 10 000 | 0.3 % | 1.84 | π | 8.45 | 1.000 | 36 % | 4.0, 11.4 | 7.8 % |

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

The theory holds until the cart reaches the rail, which happens at a median of 4.3 s. After that, the rail impacts (constraint forces up to about 40 N in the recorded data) add energy faster. Even so, the pole never rises above horizontal in any R episode: the largest $\cos\theta$ is −0.229 in R_train. So the upright region has **zero support** in the random data, as the problem statement anticipated. A second dataset is needed.

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

## 5. Unforced, undamped pool (U): ground truth for the energy-drift test

`model.dof_damping[:] = 0` and $u = 0$, so the true energy is conserved. Initial states come in three groups of 10, with the cart velocity chosen so that $p_x = 0$. The system's COM therefore stays fixed, and the cart moves at most 9 cm, so the rail is never reached.

| group | initial state | internal energy / $ag$ | MuJoCo's own max $\lvert\Delta H\rvert/(ag)$ over 10 s |
|---|---|---|---|
| small | $\varphi_0 \in [-0.5, 0.5]$, $\dot\theta_0 \in [-1, 1]$ (inside R support) | 0.01–0.11 | 5.1×10⁻⁹ |
| large | released from rest, $\cos\theta_0 \in [-0.5, 0.95]$ | 0.52–1.93 | 8.5×10⁻⁸ |
| rotating | at the bottom, $\lvert\dot\theta_0\rvert \in [8.5, 11]$ rad/s | 2.58–3.91 | 1.1×10⁻⁶ |

**Group boundaries.** The separatrix lies at internal energy $2ag$. At the bottom it corresponds to $\lvert\dot\theta\rvert = \sqrt{4ag/J_{eff,0}} = 7.81$ rad/s, so the "large" group stays below it and the "rotating" group lies above it.

**Integrator floor.** MuJoCo's drift matches our RK4 floor (PART1 §5). Any learned-model drift above about $10^{-6}\,ag$ comes from the model.

## 6. Out-of-range (OOR) test sets

The boundaries come from the training data, not from a guess. They are computed on R_train: $\max\cos\theta = -0.229$ and $\max\lvert\dot\theta\rvert = 5.29$ rad/s. Rows from E_test where the rail limit is active are excluded.

| set | definition | E_test rows | what it tests |
|---|---|---|---|
| OOR-angle | $\cos\theta > 0$ (pole above horizontal) | 3 596 | Extrapolation of $M(\theta)$ and $V(\theta)$ to angles never seen. DeLaN's networks must extrapolate here too; the structure alone does not guarantee it. |
| OOR-velocity | $\cos\theta < -0.8$ and $\lvert\dot\theta\rvert > 5.29$ | 1 046 | Extrapolation in velocity at familiar angles. DeLaN's $\ddot q$ is exactly quadratic in $\dot q$ (Coriolis) and linear in $\tau$, so the structure should carry it. The MLP has no such guarantee. |
| in R support | $\cos\theta \le -0.229$ and $\lvert\dot\theta\rvert \le 5.29$ | 3 677 | Control set: E-pool states inside the R support. |

Separating angle from velocity is deliberate. A single "upright" number would mix extrapolation that the Lagrangian structure guarantees (in $\dot q$ and $\tau$) with extrapolation that only the networks can provide (in $\theta$).

## 7. Training regimes (used in Phase 3)

- **A — random only.** $N \in \{1, 5, 20\}$ R_train episodes. For each training seed, the $N$ episodes are a random subset drawn with that seed, so the reported mean ± std covers both data variability and initialisation. Regime A is used for one-step error, rollouts, drift, OOR and data efficiency.
- **B — random + pumping.** $N$ R_train plus $N$ E_train episodes. $N = 20$ is the main model for MPPI, so every model has seen upright states and the closed-loop comparison is fair.
- **Validation.** Early stopping uses R_val in regime A and R_val + E_val in regime B. Test sets are never touched during training or model selection.
