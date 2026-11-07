# Part 1 — Analytical Euler–Lagrange model of the dm_control cart-pole

Code: [cartpole/params.py](../cartpole/params.py), [cartpole/analytical.py](../cartpole/analytical.py), [cartpole/integrators.py](../cartpole/integrators.py)
Validation: `python scripts/p1_validate.py` → `results/p1_validation.json`, `figures/p1_residuals.*`, `figures/p1_rk4_vs_mujoco.*`
Tests: `pytest -q tests/test_analytical.py`

## 1. System and coordinates

The domain is `cartpole`, task `swingup` (dm_control 1.0.51, MuJoCo 4.0.0).

- **Generalized coordinates:** $q = (x, \theta)$.
  - $x$ is the slider joint, i.e. the cart position along the rail.
  - $\theta$ is the hinge joint `hinge_1`, about the world $+y$ axis.
- **Angle convention:** the pole geometry runs `fromto 0 0 0 → 0 0 1`, so $\theta = 0$ is **upright** and $\theta = \pi$ is **hanging**. Swing-up episodes start at $\theta \approx \pi$.
- **Pole COM position:** a rotation by $\theta$ about $+y$ maps $\hat z \mapsto (\sin\theta, 0, \cos\theta)$. The pole's COM is therefore at
  $$p_{com} = (x + l\sin\theta,\; z_{cart} + l\cos\theta).$$
- **Actuation:** a single motor drives the slider, with `gear = 10` and `ctrlrange = [-1, 1]`. The generalized force is
  $$\tau = \begin{bmatrix} 10\,u \\ 0 \end{bmatrix}, \qquad u \in [-1, 1].$$
  The hinge is **unactuated**. MuJoCo's `qfrc_actuator` holds exactly this vector, so the underactuation is already part of the data.

## 2. Parameters (every value is read from `physics.model`)

| symbol | value | MuJoCo source |
|---|---|---|
| $m_c$ | 1.0 kg | `body_mass[cart]` |
| $m_p$ | 0.1 kg | `body_mass[pole_1]` |
| $l$ | 0.5 m | `body_ipos[pole_1]` relative to the hinge, perpendicular to the axis |
| $I_c$ | 0.0094246 kg m² | `body_inertia[pole_1]`, rotated by `body_iquat` and projected on the hinge axis |
| $J = I_c + m_p l^2$ | 0.0344246 kg m² | parallel-axis theorem |
| $b_x$, $b_\theta$ | 5×10⁻⁴ N s/m, 2×10⁻⁶ N m s/rad | `dof_damping` |
| armature, frictionloss | 0, 0 | `dof_armature`, `dof_frictionloss` (asserted to be 0 in code) |
| gear | 10 | `actuator_gear` |
| $g$ | 9.81 m/s² | `opt.gravity` |
| $h$ | 0.01 s, RK4 | `opt.timestep`, `opt.integrator` |
| rail | $\lvert x\rvert \le 1.8$ m, soft limit (solref 0.08, 1) | `jnt_range`, `jnt_solref` |
| flags | contacts disabled, energy enabled | `opt.disableflags = 16`, `opt.enableflags = 2` |

**Capsule inertia (check 7).** The pole is a capsule: a cylinder of length $L = 1$ and radius $r = 0.045$ with two hemispherical caps, all at uniform density. Its mass splits between the parts by volume:
$$m_{cyl} = m\frac{\pi r^2 L}{\pi r^2 L + \tfrac43\pi r^3}, \qquad m_h = \tfrac12 (m - m_{cyl}).$$
The inertia about a perpendicular axis through the COM is
$$I_c = m_{cyl}\frac{3r^2 + L^2}{12} + 2\Big[\big(\tfrac25 - \tfrac{9}{64}\big)m_h r^2 + m_h\big(\tfrac L2 + \tfrac{3r}{8}\big)^2\Big].$$
Here $\tfrac{3r}{8}$ is the distance from a hemisphere's flat face to its COM, and $\tfrac25 - \tfrac9{64}$ shifts the hemisphere's inertia about its base to its own COM. This gives $I_c = 0.0094246$, identical to MuJoCo's value (relative error 0).

**Thin-rod variant.** A textbook derivation would treat the pole as a thin rod, $I_c = mL^2/12$. That gives $J = 0.03333$, which is **3.17 % low** because it ignores the radius and the caps. We keep this "Analytical (thin rod)" model as a second analytical baseline: a hand-derived model with a realistic structural parameter error, like the URDF-vs-MuJoCo gaps found in A2.

## 3. Lagrangian and equations of motion

The COM velocity is $\dot p_{com} = (\dot x + l\cos\theta\,\dot\theta,\; -l\sin\theta\,\dot\theta)$. The pole rotates at rate $\dot\theta$.

$$T = \tfrac12 m_c \dot x^2 + \tfrac12 m_p\lvert\dot p_{com}\rvert^2 + \tfrac12 I_c\dot\theta^2
= \tfrac12 (m_c + m_p)\dot x^2 + m_p l\cos\theta\,\dot x\dot\theta + \tfrac12 J\dot\theta^2$$

$$V = m_p g l\cos\theta + \underbrace{(m_c + m_p) g\, z_{cart}}_{\text{constant; MuJoCo includes it}}$$

Write $a = m_p l$ and $M_t = m_c + m_p$. Applying $\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \tau - D\dot q$ gives two equations.

- **$x$ row:** $\frac{\partial L}{\partial \dot x} = M_t\dot x + a\cos\theta\,\dot\theta$, which gives
  $$M_t\ddot x + a\cos\theta\,\ddot\theta - a\sin\theta\,\dot\theta^2 = \tau_x - b_x\dot x.$$
- **$\theta$ row:** $\frac{\partial L}{\partial\dot\theta} = a\cos\theta\,\dot x + J\dot\theta$ and $\frac{\partial L}{\partial\theta} = -a\sin\theta\,\dot x\dot\theta + a g\sin\theta$. The $\dot x\dot\theta$ terms cancel, leaving
  $$a\cos\theta\,\ddot x + J\ddot\theta - a g\sin\theta = -b_\theta\dot\theta.$$

In matrix form, $M(q)\ddot q + C(q,\dot q)\dot q + g(q) + D\dot q = \tau$, with

$$M = \begin{bmatrix} M_t & a\cos\theta \\ a\cos\theta & J\end{bmatrix},\quad
C = \begin{bmatrix} 0 & -a\sin\theta\,\dot\theta \\ 0 & 0\end{bmatrix},\quad
g = \begin{bmatrix} 0 \\ -a g\sin\theta\end{bmatrix},\quad
D = \begin{bmatrix} b_x & 0 \\ 0 & b_\theta\end{bmatrix}.$$

**Coriolis matrix.** $C$ comes from the Christoffel symbols $c_{ijk} = \tfrac12(\partial_k M_{ij} + \partial_j M_{ik} - \partial_i M_{jk})$. The only non-zero derivative of $M$ is $\partial_\theta M_{12} = -a\sin\theta$, so the only non-zero entry is $C_{12} = c_{122}\dot\theta = -a\sin\theta\,\dot\theta$.

## 4. Structural properties used later

**(P1) Positive definiteness and scale.**
- $\det M = M_t J - a^2\cos^2\theta \ge M_t J - a^2 = 0.0354 > 0$, so $M \succ 0$ for every $\theta$.
- At $\cos\theta = \pm1$, the eigenvalues are $\lambda \in \{0.0321,\, 1.1023\}$, so the condition number is about 34.
- **Consequence for DeLaN:** take $\varepsilon = 10^{-3}$ in $M = LL^\top + \varepsilon I$. This is 30× below $\lambda_{min}$, so the network can represent $M - \varepsilon I \succ 0$ without bias. It also bounds $\lVert M^{-1}\rVert \le 1000$ during training. The two diagonal entries differ by about 30×, so the Cholesky diagonal is initialised at the right scale for each.

**(P2) Translational symmetry (Noether).** $L$ does not depend on $x$, so $x$ is cyclic and $M$ and $V$ depend on $\theta$ alone. Its conjugate momentum
$$p_x = \partial L / \partial\dot x = M_t\dot x + a\cos\theta\,\dot\theta$$
satisfies $\dot p_x = \tau_x - b_x\dot x$. It is conserved when the motion is unforced and undamped.
- **Consequence for DeLaN:** the M- and V-networks take $(\sin\theta, \cos\theta)$ only.
- $p_x$ serves as a second invariant in the drift test.

**(P3) Passivity and energy balance.**
- **Skew-symmetry for this system:** $\dot M = \begin{bmatrix} 0 & -a\sin\theta\,\dot\theta \\ -a\sin\theta\,\dot\theta & 0\end{bmatrix}$, so $\dot M - 2C = \begin{bmatrix} 0 & a\sin\theta\,\dot\theta \\ -a\sin\theta\,\dot\theta & 0\end{bmatrix}$, which is skew-symmetric.
- **Energy balance.** Let $H = \tfrac12\dot q^\top M\dot q + V$. Then
  $$\dot H = \dot q^\top M\ddot q + \tfrac12\dot q^\top\dot M\dot q + g^\top\dot q = \dot q^\top\tau - \dot q^\top D\dot q + \tfrac12\dot q^\top(\dot M - 2C)\dot q = \dot q^\top\tau - \dot q^\top D\dot q.$$
- **The identity holds for any $M(q)$, which is what DeLaN relies on.** Suppose the Coriolis term is *defined*, as DeLaN defines it, by
  $$C\dot q = \dot M\dot q - \tfrac12\partial_q(\dot q^\top M\dot q).$$
  At fixed $\dot q$, $\sum_k \dot q_k\,\partial_{q_k}(\dot q^\top M\dot q) = \dot q^\top\dot M\dot q$. Hence $\dot q^\top C\dot q = \tfrac12\dot q^\top\dot M\dot q$, and so $\dot q^\top(\dot M - 2C)\dot q = 0$ for **every** smooth $M(q)$ and $V(q)$.
- **Consequence:** with $\tau = 0$ and $D = 0$, a DeLaN conserves its own energy by construction, even when untrained. The only drift left comes from the integrator.

**(P4) Linearisations.** The cart is free, which contributes a double eigenvalue at 0.
- **Hanging**, $\theta = \pi + \varphi$: eliminating $\ddot x = (a/M_t)\ddot\varphi$ from the $x$ row gives
  $$\ddot\varphi = -\omega^2\varphi, \qquad \omega^2 = \frac{a g M_t}{J M_t - a^2}.$$
  So $\omega = 3.906$ rad/s and the period is 1.609 s.
- **Upright:** the sign flips, giving eigenvalues $\pm 3.906$ s⁻¹. Errors double every $\ln 2/\omega = 0.177$ s and grow about $e^{7.8} \approx 2400\times$ in 2 s.
- **Consequence:** 2 s rollouts that pass near upright will diverge for any imperfect model. Rollout errors are therefore read against an analytical model with 1 % parameter error.
- **Check:** the Jacobian eigenvalues computed by autograd agree to 10⁻⁵. The damping adds −4.5×10⁻⁴ and −4.7×10⁻⁵ to the real parts.

**(P5) Damping is negligible.** The time constants are $M_t/b_x = 2200$ s and $J/b_\theta = 17\,212$ s, both far longer than a 10 s episode. A learned $D$ is therefore practically unidentifiable, and an MLP's implicit damping cannot meaningfully distort the 10 s drift test.

**(P6) Rail limit.** The slider limit is a soft constraint. When $\lvert x\rvert > 1.8$, MuJoCo adds a constraint force `qfrc_constraint` of up to 39 N in the recorded data. That force is an extra generalized force, so the correct balance is
$$M\ddot q + C\dot q + g + D\dot q = \tau_{act} + \tau_{constraint}.$$
**Consequence:** training targets use $\tau_{gen} = $ `qfrc_actuator + qfrc_constraint`, rollout tests exclude windows with an active limit, and MPPI keeps $\lvert x\rvert < 1.6$.

## 5. Validation against MuJoCo (`scripts/p1_validate.py`)

**Data:** 6000 recorded states. Five episodes use random actions $u \sim U[-1,1]$, which hit the rail limit on 369 rows. One episode uses a resonant sinusoid $u = 0.3\sin\omega t$, with $\theta \in [2.26, 4.44]$, a large nonlinear swing.

| check | compared quantity | max abs residual (limit inactive) |
|---|---|---|
| 1 | $M(q)$ vs `mj_fullM` | 2.8×10⁻¹⁷ |
| 2 | $C\dot q + g$ vs `qfrc_bias` | 4.4×10⁻¹⁶ |
| 3 | $-D\dot q$ vs `qfrc_passive` | 0 |
| 4 | $\ddot q$ vs `qacc` | 1.4×10⁻¹⁴ (signal ≈ 50) |
| 5 | $\hat\tau = M\ddot q + C\dot q + g + D\dot q$ vs `mj_inverse` | 1.8×10⁻¹⁵ |
| — | $T + V$ vs `data.energy` | 3.6×10⁻¹⁵ |
| 6 | our RK4 rollout vs MuJoCo's RK4, up to 6.2 s | $\lvert\Delta x\rvert \le 2.4\times10^{-15}$, $\lvert\Delta\theta\rvert \le 1.8\times10^{-15}$ |
| 7 | capsule formula vs `body_inertia` | 0 |

**Rows with the rail limit active.** Checks 1–3 and 5 stay at round-off. Check 4 shows $2.1\times10^{-4}$ m/s², about $4\times10^{-6}$ relative. This comes from MuJoCo's constraint solver (Newton, tolerance $10^{-8}$), which stops before full convergence. Rerunning the same episode with tolerance $10^{-15}$ reduces the residual to $2.1\times10^{-14}$ (check 4b). The model is exact; the residual is solver tolerance.

**What check 6 establishes.** Our batched RK4 with zero-order-hold force reproduces MuJoCo's integrator to machine precision. In every later comparison, any error in a learned model's rollout therefore comes from the model, not the integrator. (In A2, the integrator mismatch was a visible error term.)

**Integrator floor for the energy-drift test.** This is the analytical model with $D = 0$ and $\tau = 0$, over 10 s at $h = 0.01$:

| initial condition | $\max\lvert\Delta H\rvert / (m_p g l)$ | $\max\lvert\Delta p_x\rvert$ |
|---|---|---|
| small swing (0.3 rad) | 2.2×10⁻⁹ | 1.3×10⁻¹⁰ |
| large swing | 7.2×10⁻⁹ | 1.2×10⁻⁹ |
| rotating ($\dot\theta_0 = 8$ rad/s) | 1.2×10⁻⁶ | 2.7×10⁻⁷ |

**Why the drift scales as about $h^5$.** Halving $h$ cuts the drift by 27–32×, close to $2^5$. The reason is RK4's stability function on an oscillatory mode. With $R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$,
$$\lvert R(i\omega h)\rvert^2 = 1 - \frac{(\omega h)^6}{72} + O(h^8).$$
Each step removes $O(h^6)$ of energy, and a fixed time span takes $1/h$ steps, so the total drift is $O(h^5)$. Any drift a learned model shows above this floor comes from the model, not the integrator.
