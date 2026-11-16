# Part 1 — Analytical Euler–Lagrange model of the dm_control cart-pole

- **Code:** [cartpole/params.py](../cartpole/params.py), [cartpole/analytical.py](../cartpole/analytical.py), [cartpole/integrators.py](../cartpole/integrators.py)
- **Validation:** `python scripts/p1_validate.py` → `results/p1_validation.json`, `figures/p1_residuals.*`, `figures/p1_rk4_vs_mujoco.*`
- **Tests:** `pytest -q tests/test_analytical.py`

The aim of this part is an analytical model, derived from first principles, that reproduces the simulator exactly. It serves three purposes:
- it is the "analytical" baseline;
- it is a reference for the learned models;
- it shows that every later error comes from learning, not from a modelling or integration mismatch.

---

## 0. Notation

Each symbol is also defined in the text where it first appears.

| symbol | meaning | unit |
|---|---|---|
| $x$ | cart position along the rail (slider joint) | m |
| $\theta$ | pole angle (hinge joint); $\theta = 0$ upright, $\theta = \pi$ hanging | rad |
| $q = (x, \theta)^\top$ | generalized coordinates | — |
| $\dot q = (\dot x, \dot\theta)^\top$, $\ddot q$ | generalized velocities and accelerations (a dot is $d/dt$) | — |
| $\varphi = \theta - \pi$ | angle measured from hanging (used in §4) | rad |
| $m_c$, $m_p$ | cart mass, pole mass | kg |
| $l$ | distance from the hinge to the pole's centre of mass (COM) | m |
| $r$, $\ell$ | capsule radius, length of the capsule's cylindrical part | m |
| $I_c$ | pole moment of inertia about the axis through its COM, parallel to the hinge | kg m² |
| $J = I_c + m_p l^2$ | pole moment of inertia about the hinge | kg m² |
| $M_t = m_c + m_p$ | total mass | kg |
| $a = m_p l$ | pole mass × COM distance (the coupling constant) | kg m |
| $g$ | gravitational acceleration (a scalar, 9.81) | m/s² |
| $b_x$, $b_\theta$ | viscous damping coefficients of the slider and the hinge | N s/m, N m s/rad |
| $u$ | control input (action), $u \in [-1, 1]$ | — |
| $\tau = (\tau_x, \tau_\theta)^\top$ | generalized forces (a force on $x$, a torque on $\theta$) | N, N m |
| $T$, $V$, $\mathcal L = T - V$ | kinetic energy, potential energy, Lagrangian | J |
| $H = T + V$ | total mechanical energy | J |
| $M(q)$ | mass (inertia) matrix, $2\times2$ | — |
| $C(q, \dot q)$ | Coriolis/centripetal matrix | — |
| $\mathbf g(q) = \partial V/\partial q$ | gravity vector (**bold**, unlike the scalar $g$) | N, N m |
| $D = \mathrm{diag}(b_x, b_\theta)$ | damping matrix | — |
| $h$ | integration time step | s |
| $\omega$ | small-oscillation frequency about hanging | rad/s |
| $\lambda$ | eigenvalue of the linearised system | 1/s |
| $\mu_{min}$, $\mu_{max}$ | smallest and largest eigenvalues of $M$ | kg, kg m² |

**Reserved for later parts.** $L$ (no script) is the Cholesky factor in DeLaN, $M = LL^\top + \varepsilon I$. To avoid a clash, the Lagrangian is always written $\mathcal L$, and the capsule length is $\ell$.

---

## 1. System, frames and coordinates

The domain is `cartpole` with task `swingup`, run with dm_control 1.0.51 and MuJoCo 4.0.0.

### 1.1 World frame

MuJoCo's world frame is right-handed:
- $+x$ is horizontal, along the rail;
- $+z$ is vertical, up. Gravity is `opt.gravity` $= (0, 0, -9.81)$, so it acts along $-z$ with magnitude $g = 9.81$ m/s².
- $+y = \hat z \times \hat x$ completes the frame, pointing into the screen when viewed from the default camera.

All motion happens in the $x$–$z$ plane.

### 1.2 Generalized coordinates and degrees of freedom

The model has two joints:
- **`slider`** (type *slide*, axis $+x$). It moves the cart along the rail. Its position is $x$ = `qpos[0]` and its velocity is $\dot x$ = `qvel[0]`.
- **`hinge_1`** (type *hinge*, axis $+y$, located at the cart's origin). It rotates the pole relative to the cart. Its angle is $\theta$ = `qpos[1]` and its rate is $\dot\theta$ = `qvel[1]`.

**Why two coordinates are enough.** A free rigid body moving in a plane has 3 degrees of freedom (DOF): two translations and one rotation. Two free bodies would have 6.
- The slider removes 2 of the cart's 3 (vertical motion and rotation), leaving 1.
- The hinge pins one point of the pole to the cart, removing 2 more (the pole's two translations relative to the cart), leaving its rotation.

That makes $6 - 2 - 2 = 2$ DOF in total, so $q = (x, \theta)$ is a minimal set of **generalized coordinates**: independent numbers that fix the configuration completely.

**Angles are not wrapped.** MuJoCo accumulates $\theta$ continuously over full turns. Every function of $\theta$ in the model uses $\sin\theta$ and $\cos\theta$, which are $2\pi$-periodic, so this is harmless. When angles are *compared*, the difference is wrapped to $(-\pi, \pi]$ with $\mathrm{wrap}(\alpha) = \operatorname{atan2}(\sin\alpha, \cos\alpha)$.

### 1.3 Pole orientation: the rotation matrix

The pole geometry is a capsule with `fromto = 0 0 0 → 0 0 1`, so in the pole's own frame it points along $+\hat z$. A hinge rotation by $\theta$ about $+y$ is the matrix

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}.$$

This is the standard rotation about the $y$ axis. By the right-hand rule, with the thumb along $+y$, the fingers curl from $+z$ towards $+x$. Applied to the pole axis,

$$R_y(\theta)\,\hat z = \begin{bmatrix} \sin\theta \\ 0 \\ \cos\theta \end{bmatrix}.$$

So:
- at $\theta = 0$ the pole points straight up, which is why **$\theta = 0$ is upright**;
- at $\theta = \pi$ it points straight down, **hanging**;
- a small positive $\theta$ moves the pole's tip towards $+x$.

The swing-up task starts hanging, with $\theta \approx \pi$.

### 1.4 Positions of the two bodies

From the model:
- The **cart** body is at `body_pos[cart]` $= (0, 0, z_{cart})$ with $z_{cart} = 1$ m, plus the slider displacement $x$. The cart's COM is at its origin (`body_ipos[cart]` $= 0$), so its COM is at $(x,\; z_{cart})$ in $(x, z)$ components.
- The **pole** body has its origin at the hinge, which is the cart's origin (`body_pos[pole_1]` $= 0$, `jnt_pos` $= 0$). Its COM is at `body_ipos[pole_1]` $= (0, 0, l)$ in the pole frame, with $l = 0.5$ m: halfway along the capsule, by symmetry.

Rotating that offset with §1.3 and adding the hinge position gives the **pole COM** in world coordinates:

$$p_{com} = \begin{bmatrix} x + l\sin\theta \\ z_{cart} + l\cos\theta \end{bmatrix} \quad (\text{the } x \text{ and } z \text{ components; the } y \text{ component is } 0).$$

### 1.5 Actuation and generalized forces

A single motor actuator, `slide`, acts on the slider. It has `gear = 10`, `ctrlrange = [-1, 1]` and `ctrllimited = true`, so it applies a horizontal force of $10u$ newtons to the cart, with $u$ clipped to $[-1, 1]$.

**Generalized force.** The generalized force $\tau_i$ on coordinate $q_i$ is defined through **virtual work**: for any small virtual displacement $\delta q$, the work done by the applied forces is $\delta W = \sum_i \tau_i\,\delta q_i$.
- The motor force $10u$ acts along $x$, so $\delta W = 10u\,\delta x$, giving $\tau_x = 10u$.
- No applied force or torque does work under a change $\delta\theta$, giving $\tau_\theta = 0$.

$$\tau = \begin{bmatrix} 10u \\ 0 \end{bmatrix}.$$

The hinge is **unactuated**: the pole can only be moved through its coupling to the cart (§3.8). MuJoCo stores exactly this vector in `data.qfrc_actuator`, so the underactuation is already built into the recorded data.

---

## 2. Parameters (every value is read from `physics.model`)

### 2.1 Table

| symbol | value | MuJoCo source |
|---|---|---|
| $m_c$ | 1.0 kg | `body_mass[cart]` |
| $m_p$ | 0.1 kg | `body_mass[pole_1]` |
| $l$ | 0.5 m | `body_ipos[pole_1]` − `jnt_pos[hinge_1]`, component perpendicular to the hinge axis |
| $I_c$ | 0.0094246 kg m² | `body_inertia[pole_1]`, rotated by `body_iquat` and projected on the hinge axis (§2.2) |
| $J = I_c + m_p l^2$ | 0.0344246 kg m² | parallel-axis theorem (§3.2) |
| $M_t = m_c + m_p$ | 1.1 kg | derived |
| $a = m_p l$ | 0.05 kg m | derived |
| $b_x$ | 5×10⁻⁴ N s/m | `dof_damping[0]` |
| $b_\theta$ | 2×10⁻⁶ N m s/rad | `dof_damping[1]` |
| armature, frictionloss | 0, 0 | `dof_armature`, `dof_frictionloss` (asserted to be 0 in code; if not, the model would need extra terms) |
| gear | 10 | `actuator_gear` |
| $g$ | 9.81 m/s² | `-opt.gravity[2]` |
| $h$ | 0.01 s; integrator RK4 | `opt.timestep`, `opt.integrator` |
| $z_{cart}$ | 1 m | `body_pos[cart][2]` |
| rail | $\lvert x\rvert \le 1.8$ m, soft limit | `jnt_range[slider]`, `jnt_solref` = (0.08, 1) (§4, P6) |
| flags | contacts disabled, energy enabled | `opt.disableflags = 16 = mjDSBL_CONTACT`, `opt.enableflags = 2 = mjENBL_ENERGY` |

Contacts are disabled, so the only constraint force in the whole model is the rail limit. The energy flag makes MuJoCo compute `data.energy` = (potential, kinetic), which we compare against in §5.

### 2.2 Extracting $l$ and $I_c$ from MuJoCo

- **Principal axes.** MuJoCo stores each body's inertia as three **principal moments** `body_inertia` $= (I_1, I_2, I_3)$, about principal axes rotated from the body frame by the quaternion `body_iquat`. For the pole these are $(0.0094246,\; 0.0094246,\; 0.0001001)$ with `body_iquat` $= (0, 1, 0, 0)$, a 180° rotation about $x$. That rotation swaps the signs of the $y$ and $z$ axes but not the axes themselves.
- **Inertia about the hinge axis.** The code builds the full body-frame tensor $I_{body} = R\,\mathrm{diag}(I_1, I_2, I_3)\,R^\top$, where $R$ is the rotation matrix of `body_iquat`. It then projects onto the hinge's unit axis $n$ = `jnt_axis` = $(0, 1, 0)$:
  $$I_c = n^\top I_{body}\, n.$$
  This is the moment of inertia about the line through the COM parallel to the hinge. The rule works for any orientation, so nothing depends on how MuJoCo happened to order the principal axes.
- **COM distance.** $l$ is the length of the COM offset from the hinge, with any component along the hinge axis removed:
  $$l = \lVert r - (r\cdot n)\,n\rVert, \qquad r = \texttt{body\_ipos} - \texttt{jnt\_pos}.$$

### 2.3 Capsule inertia by hand (check 7)

The pole is a **capsule**: a solid cylinder of radius $r = 0.045$ m and length $\ell = 1$ m, with a solid hemisphere of radius $r$ on each end, all of uniform density. MuJoCo derives the inertia from this geometry, scaling the density so the total mass is $m_p = 0.1$ kg. We repeat that calculation.

**Step 1 — split the mass by volume.** Uniform density means each part's mass is proportional to its volume:

$$V_{cyl} = \pi r^2 \ell = 6.3617\times10^{-3}\ \text{m}^3, \qquad V_{sph} = \tfrac43\pi r^3 = 3.8170\times10^{-4}\ \text{m}^3 \ \ (\text{both caps together form one sphere}),$$

$$m_{cyl} = m_p\frac{V_{cyl}}{V_{cyl} + V_{sph}} = 0.094340\ \text{kg}, \qquad m_h = \tfrac12(m_p - m_{cyl}) = 0.0028302\ \text{kg (each hemisphere)}.$$

**Step 2 — cylinder about a perpendicular axis through its centre.** Slice the cylinder into thin disks. Each disk contributes $\tfrac14\,dm\,r^2$ about its own diameter and $dm\,z^2$ from its offset $z$ along the axis (parallel-axis theorem). Integrating over $z \in [-\ell/2, \ell/2]$ gives

$$I_{cyl} = m_{cyl}\Big(\frac{r^2}{4} + \frac{\ell^2}{12}\Big) = m_{cyl}\frac{3r^2 + \ell^2}{12} = 4.78\times10^{-5} + 7.8616\times10^{-3} = 7.9094\times10^{-3}\ \text{kg m}^2.$$

The $\ell^2/12$ ("thin rod") part dominates. The radius adds only 0.6 %.

**Step 3 — one hemisphere about its own COM.** A solid hemisphere has the same moment of inertia about any diameter of its flat face as a full sphere per unit mass, $\tfrac25 m_h r^2$, because it is half of a sphere symmetric about that axis. Its COM lies $\tfrac{3r}{8}$ from the flat face. This follows from $\bar z = \frac{1}{V}\int z\,dV = \frac{\int_0^r z\,\pi(r^2 - z^2)\,dz}{\tfrac23\pi r^3} = \frac{\pi r^4/4}{2\pi r^3/3} = \tfrac38 r$. The parallel-axis theorem then moves the inertia from the face to the hemisphere's own COM:

$$I_{h,com} = \tfrac25 m_h r^2 - m_h\Big(\tfrac{3r}{8}\Big)^2 = \Big(\tfrac25 - \tfrac{9}{64}\Big) m_h r^2 = 1.487\times10^{-6}\ \text{kg m}^2.$$

**Step 4 — move each hemisphere to the pole's COM.** Each hemisphere's COM is at distance $d = \tfrac\ell2 + \tfrac{3r}{8} = 0.51688$ m from the pole's centre, so by the parallel-axis theorem

$$I_{h,pole} = I_{h,com} + m_h d^2 = 7.576\times10^{-4}\ \text{kg m}^2 \ \text{each}.$$

**Step 5 — sum.**

$$I_c = I_{cyl} + 2I_{h,pole} = 7.9094\times10^{-3} + 1.5152\times10^{-3} = 9.4246\times10^{-3}\ \text{kg m}^2.$$

This equals MuJoCo's value to every printed digit (relative error 0 in float64).

**Cross-check on the long axis.** About the pole's own axis, the cylinder contributes $\tfrac12 m_{cyl} r^2$ and each hemisphere contributes $\tfrac25 m_h r^2$ (a sphere's value, by symmetry). The total is $1.0010\times10^{-4}$, which equals MuJoCo's third principal moment. Both independent principal moments therefore agree with the geometry.

### 2.4 Thin-rod variant (a second analytical baseline)

A textbook derivation treats the pole as a **thin rod** of length $\ell = 1$ m, ignoring the radius and the caps: $I_c = m_p\ell^2/12 = 0.008333$ kg m², so $J = 0.03333$ kg m², which is **3.17 % low**. We keep this "Analytical (thin rod)" model as a second baseline. It is a hand-derived model with a realistic structural parameter error, like the URDF-vs-MuJoCo parameter gaps found in Assignment 2.

---

## 3. Lagrangian and equations of motion

### 3.1 Velocities

The cart's COM moves with velocity $(\dot x, 0)$, horizontally only.

The pole's COM velocity is the time derivative of $p_{com}$ (§1.4). Use the chain rule: $\frac{d}{dt}\sin\theta = \cos\theta\,\dot\theta$, $\frac{d}{dt}\cos\theta = -\sin\theta\,\dot\theta$, and $z_{cart}$ and $l$ are constant. Then

$$\dot p_{com} = \begin{bmatrix} \dot x + l\cos\theta\,\dot\theta \\ -l\sin\theta\,\dot\theta \end{bmatrix}.$$

The first component is the cart's motion plus the horizontal part of the swing. The second is the vertical part of the swing.

**Angular velocities.** The cart does not rotate (the slider allows translation only), so its angular velocity is 0. The pole's angular velocity is therefore just the hinge rate, $\omega_{pole} = \dot\theta$ about $+y$.

### 3.2 Kinetic energy

**König's theorem.** The kinetic energy of a rigid body equals the translational energy of its COM plus the rotational energy about its COM: $T_{body} = \tfrac12 m\lvert v_{com}\rvert^2 + \tfrac12 I_{com}\,\omega^2$, for planar rotation at rate $\omega$.

- **Cart:** $T_{cart} = \tfrac12 m_c\dot x^2$. There is no rotation term.
- **Pole:** $T_{pole} = \tfrac12 m_p\lvert\dot p_{com}\rvert^2 + \tfrac12 I_c\dot\theta^2$.

Expand the squared speed:

$$\lvert\dot p_{com}\rvert^2 = (\dot x + l\cos\theta\,\dot\theta)^2 + (l\sin\theta\,\dot\theta)^2
= \dot x^2 + 2l\cos\theta\,\dot x\dot\theta + l^2\cos^2\theta\,\dot\theta^2 + l^2\sin^2\theta\,\dot\theta^2
= \dot x^2 + 2l\cos\theta\,\dot x\dot\theta + l^2\dot\theta^2,$$

using $\cos^2\theta + \sin^2\theta = 1$ in the last step.

Add the two bodies and group terms by velocity product:

$$T = \tfrac12 m_c\dot x^2 + \tfrac12 m_p\big(\dot x^2 + 2l\cos\theta\,\dot x\dot\theta + l^2\dot\theta^2\big) + \tfrac12 I_c\dot\theta^2
= \tfrac12 \underbrace{(m_c + m_p)}_{M_t}\dot x^2 + \underbrace{m_p l}_{a}\cos\theta\,\dot x\dot\theta + \tfrac12\underbrace{(I_c + m_p l^2)}_{J}\dot\theta^2.$$

- **$J = I_c + m_p l^2$** is the **parallel-axis theorem**: the pole's inertia about the hinge equals its inertia about the parallel axis through its COM plus mass × (distance)².
- **The cross term $a\cos\theta\,\dot x\dot\theta$** is the cart–pole coupling.
  - It is largest when the pole is vertical: the swing velocity $l\dot\theta$ is then horizontal, parallel to the cart's velocity.
  - It vanishes at $\theta = \pm\pi/2$, with the pole horizontal: the swing velocity is then vertical, perpendicular to the cart's velocity.
  - It is the only route by which the cart force reaches the unactuated pole.

**Reading off the mass matrix.** Any kinetic energy that is quadratic in the velocities can be written $T = \tfrac12\dot q^\top M(q)\dot q$ with a symmetric $M$. Expanding the right-hand side for 2×2 gives $\tfrac12\big(M_{11}\dot x^2 + 2M_{12}\dot x\dot\theta + M_{22}\dot\theta^2\big)$, and matching coefficients gives

$$M(\theta) = \begin{bmatrix} M_t & a\cos\theta \\ a\cos\theta & J \end{bmatrix}.$$

The cross term is split equally between $M_{12}$ and $M_{21}$, which is why there is no factor 2. Equivalently, $M_{ij} = \partial^2 T/\partial\dot q_i\,\partial\dot q_j$.

### 3.3 Potential energy

Gravity's potential is $V = \sum_i m_i\,g\,z_i$, with each height $z_i$ taken at the body's COM:

$$V = m_c g\,z_{cart} + m_p g\,(z_{cart} + l\cos\theta) = \underbrace{a g\cos\theta}_{\text{depends on }\theta} + \underbrace{M_t\,g\,z_{cart}}_{\text{constant}}.$$

- **The constant term** does not affect the motion, because only the derivatives $\partial V/\partial q$ enter the equations. We keep it only so our $V$ matches MuJoCo's `data.energy[0]`, which measures every height from $z = 0$. In code, this is `AnalyticalModel.potential(..., mujoco_offset=True)`.
- **Sign check.** $V$ is largest at $\theta = 0$ (upright, the highest point) and smallest at $\theta = \pi$ (hanging).
- **Number check.** Hanging, $\cos\theta = -1$, so $V = 1.1\cdot9.81\cdot1 - 0.05\cdot9.81 = 10.791 - 0.4905 = 10.3005$ J. At the end of the first smoke-test episode, with the pole near hanging, MuJoCo reported `energy[0]` $= 10.30116$ J. The small excess, $0.0007$ J, is the pole's offset from exactly hanging. The exact, row-by-row comparison is the energy check in §5 (residual $3.6\times10^{-15}$).

### 3.4 Damping

MuJoCo's `dof_damping` applies a viscous force proportional to and opposing each joint velocity: `qfrc_passive` $= -D\dot q$, with $D = \mathrm{diag}(b_x, b_\theta)$. In Lagrangian mechanics, such forces come from the **Rayleigh dissipation function** $\mathcal R = \tfrac12\dot q^\top D\dot q$, whose generalized force is $-\partial\mathcal R/\partial\dot q = -D\dot q$. The dissipated power is $2\mathcal R = \dot q^\top D\dot q \ge 0$.

### 3.5 Euler–Lagrange equations with non-conservative forces

Hamilton's principle, extended with the virtual work of the non-conservative forces (the actuator $\tau$ and the damping $-D\dot q$), gives for each coordinate $q_i$

$$\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot q_i} - \frac{\partial\mathcal L}{\partial q_i} = \tau_i - (D\dot q)_i, \qquad \mathcal L = T - V.$$

Gravity is conservative and enters through $V$. Everything else enters on the right-hand side.

### 3.6 The $x$ equation

1. $\dfrac{\partial\mathcal L}{\partial\dot x} = M_t\dot x + a\cos\theta\,\dot\theta$. Only $T$ depends on $\dot x$.
2. $\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial\dot x} = M_t\ddot x + a\cos\theta\,\ddot\theta - a\sin\theta\,\dot\theta^2$, by the product rule on $\cos\theta\,\dot\theta$.
3. $\dfrac{\partial\mathcal L}{\partial x} = 0$. Neither $T$ nor $V$ contains $x$ (§4, P2).

So

$$M_t\ddot x + a\cos\theta\,\ddot\theta - a\sin\theta\,\dot\theta^2 = \tau_x - b_x\dot x.$$

### 3.7 The $\theta$ equation

1. $\dfrac{\partial\mathcal L}{\partial\dot\theta} = a\cos\theta\,\dot x + J\dot\theta$.
2. $\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial\dot\theta} = a\cos\theta\,\ddot x - a\sin\theta\,\dot\theta\,\dot x + J\ddot\theta$.
3. $\dfrac{\partial\mathcal L}{\partial\theta} = \dfrac{\partial T}{\partial\theta} - \dfrac{\partial V}{\partial\theta} = -a\sin\theta\,\dot x\dot\theta - (-a g\sin\theta) = -a\sin\theta\,\dot x\dot\theta + a g\sin\theta$.

Subtract (3) from (2). The two $a\sin\theta\,\dot x\dot\theta$ terms cancel, leaving

$$a\cos\theta\,\ddot x + J\ddot\theta - a g\sin\theta = \tau_\theta - b_\theta\dot\theta = -b_\theta\dot\theta.$$

**Physical reading.**
- $J\ddot\theta - ag\sin\theta$ is a pendulum about the hinge. Gravity's torque $ag\sin\theta$ pushes the pole *away* from upright, because upright is unstable.
- $a\cos\theta\,\ddot x$ is the inertial torque felt in the accelerating cart frame. Accelerating the cart is the only way to torque the pole.

### 3.8 Matrix form

Collect both rows as $M(q)\ddot q + C(q,\dot q)\dot q + \mathbf g(q) + D\dot q = \tau$:

$$M = \begin{bmatrix} M_t & a\cos\theta \\ a\cos\theta & J\end{bmatrix},\quad
C = \begin{bmatrix} 0 & -a\sin\theta\,\dot\theta \\ 0 & 0\end{bmatrix},\quad
\mathbf g = \frac{\partial V}{\partial q} = \begin{bmatrix} 0 \\ -a g\sin\theta\end{bmatrix},\quad
D = \begin{bmatrix} b_x & 0 \\ 0 & b_\theta\end{bmatrix}.$$

So $C\dot q = (-a\sin\theta\,\dot\theta^2,\; 0)^\top$.

**Physical reading of $C\dot q$.** The pole's COM moves on a circle of radius $l$ around the hinge, so it needs a centripetal force $m_p l\dot\theta^2$ directed towards the hinge. That force's horizontal component, $a\sin\theta\,\dot\theta^2$, must come through the cart. It therefore appears in the $x$ row and not in the $\theta$ row.

**Forward and inverse dynamics.** These are the two ways the equation is used:
- **Forward dynamics** solves for the acceleration: $\ddot q = M^{-1}(\tau - C\dot q - \mathbf g - D\dot q)$. This is `AnalyticalModel.qdd`, which uses a closed-form 2×2 solve (Cramer's rule).
- **Inverse dynamics** solves for the force: $\hat\tau = M\ddot q + C\dot q + \mathbf g + D\dot q$. This is `AnalyticalModel.inverse`.

### 3.9 Coriolis matrix from Christoffel symbols

The vector $C\dot q$ is unique, but the matrix $C$ is not: many matrices give the same product. We use the standard choice built from the **Christoffel symbols of the first kind**,

$$c_{ijk} = \tfrac12\Big(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\Big), \qquad C_{ij} = \sum_k c_{ijk}\,\dot q_k,$$

because it makes $\dot M - 2C$ skew-symmetric (§4, P3).

Here $M$ depends on $\theta$ only, and its only non-constant entry is $M_{12} = M_{21} = a\cos\theta$. So the only non-zero partial derivative is

$$\partial_\theta M_{12} = \partial_\theta M_{21} = -a\sin\theta.$$

Index 1 is $x$ and index 2 is $\theta$. All eight symbols:

| $c_{ijk}$ | value | | $c_{ijk}$ | value |
|---|---|---|---|---|
| $c_{111}$ | 0 | | $c_{211}$ | $\tfrac12(0 + 0 - \partial_\theta M_{11}) = 0$ |
| $c_{112}$ | $\tfrac12(\partial_\theta M_{11} + \partial_x M_{12} - \partial_x M_{12}) = 0$ | | $c_{212}$ | $\tfrac12(\partial_\theta M_{21} + \partial_x M_{22} - \partial_\theta M_{12}) = 0$ |
| $c_{121}$ | $\tfrac12(\partial_x M_{12} + \partial_\theta M_{11} - \partial_x M_{21}) = 0$ | | $c_{221}$ | $\tfrac12(\partial_x M_{22} + \partial_\theta M_{21} - \partial_\theta M_{21}) = 0$ |
| $c_{122}$ | $\tfrac12(\partial_\theta M_{12} + \partial_\theta M_{12} - \partial_x M_{22}) = -a\sin\theta$ | | $c_{222}$ | $\tfrac12\partial_\theta M_{22} = 0$ |

Only $c_{122} \ne 0$, so $C_{12} = c_{121}\dot x + c_{122}\dot\theta = -a\sin\theta\,\dot\theta$, and every other entry is 0, as written in §3.8.

---

## 4. Structural properties used later (P1–P6)

### P1. Positive definiteness and scale of $M$

- **Positive definiteness.** A symmetric 2×2 matrix is positive definite ($M \succ 0$) exactly when $M_{11} > 0$ and $\det M > 0$. Here $M_{11} = M_t > 0$, and
  $$\det M = M_t J - a^2\cos^2\theta \;\ge\; M_t J - a^2 = (m_c + m_p)(I_c + m_p l^2) - m_p^2 l^2 = m_c I_c + m_c m_p l^2 + m_p I_c > 0.$$
  Every term in the final sum is positive, so $M \succ 0$ for **every** $\theta$. Numerically the minimum is $\det M = 0.035367$, at $\cos\theta = \pm1$. Physically, a positive-definite $M$ means any motion has positive kinetic energy.
- **Eigenvalues.** For a 2×2 symmetric matrix, the eigenvalues are
  $$\mu_{min,max} = \tfrac12\Big(\mathrm{tr}\,M \mp \sqrt{(\mathrm{tr}\,M)^2 - 4\det M}\Big).$$
  At $\cos\theta = \pm1$, with $\mathrm{tr}\,M = M_t + J = 1.13442$, this gives $\mu_{min} = 0.03208$ and $\mu_{max} = 1.10234$. At $\cos\theta = 0$, $M$ is diagonal with eigenvalues $J = 0.0344$ and $M_t = 1.1$. The **condition number** $\kappa = \mu_{max}/\mu_{min}$ is at most 34.4.
- **Consequence for DeLaN** (Part 3). DeLaN writes $M = LL^\top + \varepsilon I$, where $L$ is a lower-triangular Cholesky factor with a positive diagonal and $\varepsilon > 0$ is a small constant.
  - Since $LL^\top \succeq 0$, $\mu_{min}(M) \ge \varepsilon$, so $\lVert M^{-1}\rVert_2 = 1/\mu_{min} \le 1/\varepsilon$. This bounds the forward dynamics during training.
  - We choose $\varepsilon = 10^{-3}$: 30× below the true $\mu_{min} = 0.032$. Any true $M$ with $\mu_{min} > \varepsilon$ can be represented exactly, because $M - \varepsilon I$ is still positive definite and so has a Cholesky factor. The choice therefore introduces no bias.

### P2. Translational symmetry (Noether's theorem)

- **$x$ is cyclic.** Neither $T$ nor $V$ contains $x$, so $\partial\mathcal L/\partial x = 0$, and $x$ is called a **cyclic** (ignorable) coordinate. Physically, the rail is horizontal and gravity is vertical, so shifting the whole system along the rail changes nothing.
- **Conserved momentum.** The $x$ equation (§3.6) then reads $\frac{d}{dt}p_x = \tau_x - b_x\dot x$, with the **conjugate momentum**
  $$p_x = \frac{\partial\mathcal L}{\partial\dot x} = M_t\dot x + a\cos\theta\,\dot\theta = m_c\dot x + m_p(\dot x + l\cos\theta\,\dot\theta).$$
  The right-hand form shows that $p_x$ is the system's total horizontal linear momentum: the cart's plus the pole COM's. It is **conserved** when the motion is unforced ($\tau_x = 0$) and undamped ($b_x = 0$). This is Noether's theorem: a continuous symmetry (translation in $x$) gives a conserved quantity ($p_x$).
- **Consequences:**
  - DeLaN's $M$- and $V$-networks take $(\sin\theta, \cos\theta)$ only, not $x$.
  - $p_x$ is a second invariant, next to $H$, for the energy-drift test.
  - Choosing $p_x = 0$ in an initial state keeps the system's COM fixed. Part 2 uses this to keep the unforced cart away from the rail.

### P3. Passivity and energy balance

- **Time derivative of $M$.** $M$ depends on time only through $\theta$, so by the chain rule
  $$\dot M = \frac{\partial M}{\partial\theta}\dot\theta = \begin{bmatrix} 0 & -a\sin\theta\,\dot\theta \\ -a\sin\theta\,\dot\theta & 0\end{bmatrix}.$$
- **Skew-symmetry.** With $C$ from §3.9,
  $$\dot M - 2C = \begin{bmatrix} 0 & -a\sin\theta\,\dot\theta + 2a\sin\theta\,\dot\theta \\ -a\sin\theta\,\dot\theta & 0\end{bmatrix} = \begin{bmatrix} 0 & a\sin\theta\,\dot\theta \\ -a\sin\theta\,\dot\theta & 0\end{bmatrix}.$$
  This is **skew-symmetric** ($S^\top = -S$), so $v^\top(\dot M - 2C)v = 0$ for every vector $v$.
- **Energy balance.** Let $H = T + V = \tfrac12\dot q^\top M\dot q + V$. Differentiate term by term:
  - $\frac{d}{dt}\big(\tfrac12\dot q^\top M\dot q\big) = \dot q^\top M\ddot q + \tfrac12\dot q^\top\dot M\dot q$, using the symmetry of $M$;
  - $\frac{d}{dt}V = \frac{\partial V}{\partial q}\dot q = \mathbf g^\top\dot q$.

  Now substitute the equation of motion, $M\ddot q = \tau - C\dot q - \mathbf g - D\dot q$:
  $$\dot H = \dot q^\top(\tau - C\dot q - \mathbf g - D\dot q) + \tfrac12\dot q^\top\dot M\dot q + \mathbf g^\top\dot q = \dot q^\top\tau - \dot q^\top D\dot q + \tfrac12\dot q^\top(\dot M - 2C)\dot q = \underbrace{\dot q^\top\tau}_{\text{power in}} - \underbrace{\dot q^\top D\dot q}_{\text{dissipated}\ \ge 0}.$$

  The gravity terms cancel, and the last bracket vanishes by skew-symmetry. With no input and no damping, $H$ is exactly conserved. This energy balance is called **passivity**.
- **The identity holds for ANY $M(q)$, which is what DeLaN relies on.** DeLaN does not compute $C$ from Christoffel symbols. It *defines* the Coriolis vector directly from a learned $M$:
  $$C\dot q \;:=\; \dot M\dot q - \tfrac12\,\partial_q\big(\dot q^\top M\dot q\big),$$
  where $\partial_q(\dot q^\top M\dot q)$ is the vector with components $\dot q^\top\frac{\partial M}{\partial q_k}\dot q$, i.e. the derivative with respect to $q$ with $\dot q$ held fixed.
  - **Proof.** Take the dot product with $\dot q$:
    $$\dot q^\top\partial_q(\dot q^\top M\dot q) = \sum_k \dot q_k\,\dot q^\top\frac{\partial M}{\partial q_k}\dot q = \dot q^\top\Big(\sum_k\frac{\partial M}{\partial q_k}\dot q_k\Big)\dot q = \dot q^\top\dot M\dot q.$$
    Hence $\dot q^\top C\dot q = \dot q^\top\dot M\dot q - \tfrac12\dot q^\top\dot M\dot q = \tfrac12\dot q^\top\dot M\dot q$, so $\dot q^\top(\dot M - 2C)\dot q = 0$. Nothing about the specific $M$ or $V$ was used, so the energy balance holds for **every** smooth $M(q)$ and $V(q)$, trained or not.
  - **Consistency with §3.9.** For our $M$: $\dot M\dot q = (-a\sin\theta\,\dot\theta^2,\; -a\sin\theta\,\dot\theta\dot x)^\top$, and $\tfrac12\partial_q(\dot q^\top M\dot q) = \tfrac12(0,\; 2\cdot(-a\sin\theta)\,\dot x\dot\theta)^\top = (0,\; -a\sin\theta\,\dot x\dot\theta)^\top$. The difference is $(-a\sin\theta\,\dot\theta^2,\; 0)^\top$, the same $C\dot q$ as the Christoffel construction.
- **Consequence.** With $\tau = 0$ and $D = 0$, a DeLaN conserves its *own* energy by construction, even when untrained. The only drift left comes from the time integrator (§5.4).

### P4. Linearisations about the two equilibria

The state is $s = (x, \theta, \dot x, \dot\theta)$. Setting $\dot s = 0$ with $\tau = 0$ requires $\sin\theta = 0$, so the equilibria are hanging ($\theta = \pi$) and upright ($\theta = 0$), at rest, for **any** $x$. The equilibria form a continuum along the rail, which is why each linearisation has eigenvalues at 0 (cart position and velocity).

**Hanging.** Write $\theta = \pi + \varphi$ with small $\varphi$. Then $\cos\theta \approx -1$, $\sin\theta \approx -\varphi$, and products of small quantities ($\dot\theta^2$, $\varphi\dot\theta$, …) are dropped. With $\tau = 0$ and $D = 0$:
- $x$ row: $M_t\ddot x - a\ddot\varphi = 0$, so $\ddot x = (a/M_t)\,\ddot\varphi$.
- $\theta$ row: $-a\ddot x + J\ddot\varphi + ag\varphi = 0$, since $-ag\sin\theta = -ag(-\varphi)$.

Substituting the first into the second:

$$\underbrace{\Big(J - \frac{a^2}{M_t}\Big)}_{J_{eff,0}}\ddot\varphi + ag\,\varphi = 0 \;\Rightarrow\; \ddot\varphi = -\omega^2\varphi, \qquad \omega^2 = \frac{ag}{J_{eff,0}} = \frac{agM_t}{JM_t - a^2}.$$

Numerically $J_{eff,0} = 0.032152$ kg m² and $\omega^2 = 15.256$ s⁻², so $\omega = 3.906$ rad/s, with period $2\pi/\omega = 1.609$ s and frequency 0.62 Hz.

**Why the free cart matters.** On a *fixed* pivot, $\omega_0 = \sqrt{ag/J} = 3.775$ rad/s. The free cart recoils as the pole swings, which lowers the effective inertia from $J$ to $J_{eff,0}$ and raises the frequency by 3.5 %.

**Upright.** Write $\theta = \varphi$ with small $\varphi$. Then $\cos\theta \approx 1$ and $\sin\theta \approx \varphi$, and the same elimination gives
$$J_{eff,0}\ddot\varphi - ag\varphi = 0.$$
The solutions are $e^{\pm\lambda t}$ with $\lambda = \omega = 3.906$ s⁻¹: the upright is a saddle, unstable.
- **Error growth.** A small state error grows like $e^{\lambda t}$. It doubles every $\ln 2/\lambda = 0.177$ s and grows $e^{2\lambda} \approx 2470\times$ in 2 s.
- **Consequence.** 2 s rollouts that pass near upright will diverge for any imperfect model. Rollout errors are therefore read against an analytical model with 1 % parameter error.

**Check.** The 4×4 Jacobian of $\dot s$, computed by autograd (`p1_validate.py`), has eigenvalues:
- hanging: $-4.7\times10^{-5} \pm 3.90586\,i$, plus $0$ and $-4.5\times10^{-4}$;
- upright: $+3.9058$ and $-3.9059$, plus $0$ and $-4.5\times10^{-4}$.

These agree with the formula to $10^{-5}$. With damping included, the cart's zero eigenvalue moves to $-b_x/M_t = -4.5\times10^{-4}$, and the pole's pair gains a real part of $-4.7\times10^{-5}$.

### P5. Damping is negligible on our time scales

**Time constants.** For the cart alone, $M_t\ddot x = -b_x\dot x$ gives $\dot x(t) = \dot x_0\,e^{-t/t_x}$ with time constant $t_x = M_t/b_x = 2200$ s. For the pole, $t_\theta = J/b_\theta = 17\,212$ s. Both are far longer than a 10 s episode.

**Energy lost in 10 s.** Kinetic energy decays like $e^{-2t/t_{x,\theta}}$, so over 10 s the cart loses at most $1 - e^{-20/2200} = 0.9\,\%$ of its kinetic energy and the pole at most $0.12\,\%$.

**Consequences:**
- The data barely constrains a learned $D$, so it is practically unidentifiable.
- An MLP, which cannot switch damping off, is at most negligibly disadvantaged in the undamped 10 s drift test.

### P6. Rail limit

The slider's range $\lvert x\rvert \le 1.8$ m is a **soft constraint**. When the cart goes past the limit, MuJoCo's constraint solver adds a force `qfrc_constraint` that behaves like a critically damped spring with time constant 0.08 s (`solref` = (time constant 0.08 s, damping ratio 1)). That force is an extra generalized force, so the correct balance on those rows is

$$M\ddot q + C\dot q + \mathbf g + D\dot q = \tau_{act} + \tau_{con}, \qquad \tau_{act} = \texttt{qfrc\_actuator}, \quad \tau_{con} = \texttt{qfrc\_constraint}.$$

**Size.** Against the actuator's ±10 N, the constraint force reaches 39 N in the 5 validation episodes, and in the Part 2 data 61 N (R_train), 67 N (E_train) and 178 N (S_train, where a falling episode hits the rail).

**Consequences:**
- The validation includes $\tau_{con}$ on limit rows.
- Training drops those rows (Part 2, [ISSUES.md](ISSUES.md) D2).
- Rollout tests exclude windows that touch the limit.
- MPPI keeps $\lvert x\rvert < 1.6$ m.

---

## 5. Validation against MuJoCo (`scripts/p1_validate.py`)

### 5.1 Data and MuJoCo quantities compared

**Data.** 6000 recorded states:
- five episodes with random actions $u \sim U[-1,1]$, independent at every step, which hit the rail limit on 369 rows;
- one episode with a resonant sinusoid $u = 0.3\sin(\omega t)$, which swings the pole through $\theta \in [2.26, 4.44]$, a large, strongly nonlinear swing.

Each row is recorded after `mj_forward` with the row's action (Part 2 §1).

**MuJoCo quantities** (all computed by MuJoCo for the same state):

| MuJoCo field / function | meaning | our expression |
|---|---|---|
| `mj_fullM(m, d, M)` | dense mass matrix | $M(q)$ |
| `qfrc_bias` | "bias" force: everything in $M\ddot q = \dots$ that does not depend on $\ddot q$ or the inputs, computed by recursive Newton–Euler with $\ddot q = 0$ | $C\dot q + \mathbf g$ |
| `qfrc_passive` | passive forces (damping; there are no springs here) | $-D\dot q$ |
| `qacc` | forward-dynamics acceleration | $\ddot q = M^{-1}(\tau_{act} + \tau_{con} - C\dot q - \mathbf g - D\dot q)$ |
| `mj_inverse` → `qfrc_inverse` | inverse dynamics: the applied force needed to produce a given `qacc`, i.e. $M\ddot q + $ `qfrc_bias` $-$ `qfrc_passive` $-$ `qfrc_constraint` | $\hat\tau - \tau_{con}$ with $\hat\tau = M\ddot q + C\dot q + \mathbf g + D\dot q$ |
| `data.energy` | (potential, kinetic) with heights from $z = 0$ | $(V,\ T)$ including the constant of §3.3 |

The **residual** of a check is (our value) − (MuJoCo's value). The table reports its largest absolute entry over all rows. Float64 round-off is about $2.2\times10^{-16}$ relative, so residuals of order $10^{-15}$ to $10^{-14}$ on signals of order 1–50 are machine precision.

### 5.2 Results

| check | compared quantity | max abs residual (rows with limit inactive) |
|---|---|---|
| 1 | $M(q)$ vs `mj_fullM` | 2.8×10⁻¹⁷ |
| 2 | $C\dot q + \mathbf g$ vs `qfrc_bias` | 4.4×10⁻¹⁶ |
| 3 | $-D\dot q$ vs `qfrc_passive` | 0 |
| 4 | $\ddot q$ vs `qacc` | 1.4×10⁻¹⁴ (signal up to ≈ 50) |
| 5 | $\hat\tau - \tau_{con}$ vs `qfrc_inverse` | 1.8×10⁻¹⁵ |
| — | $T + V$ vs `data.energy` (sum) | 3.6×10⁻¹⁵ |
| 6 | our RK4 rollout vs MuJoCo's RK4, up to 6.2 s | $\lvert\Delta x\rvert \le 2.4\times10^{-15}$, $\lvert\Delta\theta\rvert \le 1.8\times10^{-15}$ |
| 7 | capsule formula (§2.3) vs `body_inertia` | 0 |

**What each check covers:**
- Check 1 validates $T$, since $M$ is its Hessian with respect to $\dot q$.
- Check 2 validates the Coriolis term and gravity.
- Check 3 validates the damping.
- Check 4 validates the full forward dynamics.
- Check 5 validates the inverse dynamics, as the course instructions require.

### 5.3 Rows with the rail limit active (check 4b)

Checks 1–3 and 5 stay at round-off on these rows. Check 4 shows $2.1\times10^{-4}$ m/s², about $4\times10^{-6}$ relative to the signal.

**Cause.** MuJoCo computes constraint forces with an iterative Newton solver that stops once its improvement falls below `opt.tolerance` $= 10^{-8}$. The reported `qacc` and `qfrc_constraint` are therefore consistent only to that tolerance.

**Test.** Rerunning the same episode with `tolerance` $= 10^{-15}$ (and up to 1000 iterations) reduces the residual to $2.1\times10^{-14}$, i.e. round-off. The model is exact, and the residual is solver tolerance.

### 5.4 Integration: our RK4 equals MuJoCo's (check 6)

**State and dynamics.** The state is $y = (q, \dot q)$, and its derivative is $f(y) = (\dot q,\; \ddot q(q, \dot q, \tau))$. MuJoCo's `RK4` integrator holds `ctrl`, and hence $\tau$, constant over each step: a **zero-order hold**. It then applies the classical 4th-order Runge–Kutta method with step $h$:

$$k_1 = f(y_n),\quad k_2 = f\big(y_n + \tfrac h2 k_1\big),\quad k_3 = f\big(y_n + \tfrac h2 k_2\big),\quad k_4 = f(y_n + h k_3),\qquad y_{n+1} = y_n + \tfrac h6(k_1 + 2k_2 + 2k_3 + k_4).$$

[cartpole/integrators.py](../cartpole/integrators.py) implements exactly this, batched in torch. Slide and hinge joints have positions that live in ordinary vector spaces (no quaternions), so MuJoCo's position update is the same linear combination.

**Result.** Rolling out the analytical model with the recorded actions reproduces MuJoCo's trajectory to $\le 2.4\times10^{-15}$ for up to 6.2 s (until the first rail contact).

**Why this matters.** In every later comparison, a learned model's rollout error comes from the model, not from the integrator. In Assignment 2, integrator mismatch was a visible error term.

### 5.5 The integrator's energy floor (for the drift test)

With $D = 0$ and $\tau = 0$, the exact $H$ is constant, but RK4 is not exactly energy-conserving. Its drift sets the floor below which no model's drift can be measured.

**Measured** (`results/p1_validation.json` → `integrator_energy_floor_10s`). Analytical model, 10 s; every initial state has $\dot x_0 = 0.5$ m/s:

| initial state $(\theta_0, \dot\theta_0)$ | motion | $\max_t\lvert H(t) - H(0)\rvert/(ag)$ at $h = 0.01$ | at $h = 0.005$ | ratio | $\max_t\lvert\Delta p_x\rvert$ at $h = 0.01$ |
|---|---|---|---|---|---|
| $(\pi - 0.3,\; 0)$ | swing of amplitude 0.30 rad | 2.2×10⁻⁹ | 7.5×10⁻¹¹ | 29.9 | 1.3×10⁻¹⁰ |
| $(3.0,\; 2.0)$ | swing of amplitude 0.54 rad | 7.2×10⁻⁹ | 2.6×10⁻¹⁰ | 27.2 | 1.2×10⁻⁹ |
| $(0.5,\; 8.0)$ | full rotations | 1.2×10⁻⁶ | 4.4×10⁻⁸ | 28.2 | 2.6×10⁻⁷ |

Energies are normalised by $ag = m_p g l = 0.4905$ J, the natural energy scale of the pole. The amplitude of the second swing follows from its conserved internal energy (Part 2 §3).

**Why halving $h$ divides the drift by about $2^5 = 32$.** Consider the linear oscillator $\ddot\varphi = -\omega^2\varphi$.
1. **Write it as one complex equation.** Let $z = \omega\varphi + i\dot\varphi$. Then $\dot z = \omega\dot\varphi + i\ddot\varphi = \omega\dot\varphi - i\omega^2\varphi = -i\omega z$, and the energy is proportional to $\lvert z\rvert^2 = \omega^2\varphi^2 + \dot\varphi^2$.
2. **One RK4 step multiplies $z$ by the stability function.** On $\dot z = \kappa z$, one step gives $z_{n+1} = R(h\kappa)\,z_n$ with
   $$R(w) = 1 + w + \tfrac{w^2}{2} + \tfrac{w^3}{6} + \tfrac{w^4}{24},$$
   the Taylor series of $e^w$ cut after 4th order. The energy is multiplied by $\lvert R(-i\omega h)\rvert^2 = \lvert R(i\omega h)\rvert^2$ per step; the two are equal because $R$ has real coefficients.
3. **Expand.** Let $y = \omega h$. Then $R(iy) = \big(1 - \tfrac{y^2}{2} + \tfrac{y^4}{24}\big) + i\big(y - \tfrac{y^3}{6}\big)$, and
   $$\lvert R(iy)\rvert^2 = 1 - y^2 + \tfrac{y^4}{4} + \tfrac{y^4}{12} - \tfrac{y^6}{24} + \tfrac{y^8}{576} + y^2 - \tfrac{y^4}{3} + \tfrac{y^6}{36} = 1 - \frac{y^6}{72} + \frac{y^8}{576}.$$
   The $y^2$ and $y^4$ terms cancel exactly.
4. **Accumulate over a fixed time.** Each step therefore loses a fraction $(\omega h)^6/72$ of the energy. Covering a fixed time $t$ takes $t/h$ steps, so the total relative loss is about $t\,\omega^6 h^5/72$, proportional to $h^5$.

**Comparison with the measurement.** The measured ratios of 27–30 per halving of $h$ match this $h^5$ scaling; the gap below 32 comes from the nonlinearity and the higher-order terms. The rotating case drifts most, because its fast rotation acts like a larger $\omega$.

**Consequence.** Any drift a learned model shows clearly above about $10^{-6}\,ag$ comes from the model, not from the integrator.
