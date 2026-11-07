"""Physical parameters of dm_control cartpole, read from the MuJoCo model.

Nothing here is hard-coded: every number comes from physics.model so the
analytical model is exactly the simulated system.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from dm_control import suite  # noqa: E402


def make_env(seed: int = 0, task: str = "swingup"):
    """dm_control cartpole environment with a fixed seed."""
    return suite.load("cartpole", task, task_kwargs={"random": seed})


@dataclass(frozen=True)
class CartpoleParams:
    m_c: float        # cart mass [kg]
    m_p: float        # pole mass [kg]
    l: float          # hinge -> pole COM distance [m]
    I_c: float        # pole inertia about its COM, about the hinge axis [kg m^2]
    b_x: float        # slider viscous damping [N s/m]
    b_th: float       # hinge viscous damping [N m s/rad]
    gear: float       # actuator gear: tau_x = gear * u
    g: float          # gravity magnitude [m/s^2]
    dt: float         # simulation timestep [s]
    x_limit: float    # slider range is [-x_limit, x_limit] [m]
    z_cart: float     # height of the cart body (MuJoCo potential-energy offset) [m]

    @property
    def M_tot(self) -> float:
        return self.m_c + self.m_p

    @property
    def mpl(self) -> float:
        return self.m_p * self.l

    @property
    def J(self) -> float:
        """Pole inertia about the hinge (parallel-axis theorem)."""
        return self.I_c + self.m_p * self.l ** 2

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(M_tot=self.M_tot, mpl=self.mpl, J=self.J)
        return d


def load_params(physics=None) -> CartpoleParams:
    if physics is None:
        physics = make_env().physics
    m = physics.model
    cart = m.name2id("cart", "body")
    pole = m.name2id("pole_1", "body")
    slider = m.name2id("slider", "joint")
    hinge = m.name2id("hinge_1", "joint")
    act = m.name2id("slide", "actuator")

    # Pole inertia about the hinge axis: rotate the principal inertia into the
    # body frame (body_iquat) and project onto the joint axis.
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[pole])
    R = R.reshape(3, 3)
    I_body = R @ np.diag(m.body_inertia[pole]) @ R.T
    axis = m.jnt_axis[hinge] / np.linalg.norm(m.jnt_axis[hinge])
    I_c = float(axis @ I_body @ axis)

    # COM offset from the hinge, perpendicular to the hinge axis.
    r = m.body_ipos[pole] - m.jnt_pos[hinge]
    r_perp = r - (r @ axis) * axis
    l = float(np.linalg.norm(r_perp))

    assert m.jnt_type[slider] == mujoco.mjtJoint.mjJNT_SLIDE
    assert m.jnt_type[hinge] == mujoco.mjtJoint.mjJNT_HINGE
    assert np.allclose(m.dof_armature, 0) and np.allclose(m.dof_frictionloss, 0), \
        "model has armature/friction: the analytical model would need them"
    lo, hi = m.jnt_range[slider]
    assert np.isclose(-lo, hi)

    return CartpoleParams(
        m_c=float(m.body_mass[cart]),
        m_p=float(m.body_mass[pole]),
        l=l,
        I_c=I_c,
        b_x=float(m.dof_damping[m.jnt_dofadr[slider]]),
        b_th=float(m.dof_damping[m.jnt_dofadr[hinge]]),
        gear=float(m.actuator_gear[act, 0]),
        g=float(-m.opt.gravity[2]),
        dt=float(m.opt.timestep),
        x_limit=float(hi),
        z_cart=float(m.body_pos[cart][2]),
    )


def capsule_inertia_perp(mass: float, radius: float, length: float) -> float:
    """Inertia of a solid capsule about a COM axis perpendicular to its length.

    Cylinder (length L, radius r) plus two hemispherical caps, uniform density,
    mass split by volume. Each hemisphere's COM sits 3r/8 beyond the cylinder end.
    """
    r, L = radius, length
    v_cyl = np.pi * r ** 2 * L
    v_sph = 4.0 / 3.0 * np.pi * r ** 3
    m_cyl = mass * v_cyl / (v_cyl + v_sph)
    m_hemi = 0.5 * mass * v_sph / (v_cyl + v_sph)
    I_cyl = m_cyl * (3 * r ** 2 + L ** 2) / 12.0
    # hemisphere about its own COM (perpendicular axis): (2/5 - 9/64) m r^2
    I_hemi_com = (2.0 / 5.0 - 9.0 / 64.0) * m_hemi * r ** 2
    d = L / 2.0 + 3.0 * r / 8.0
    return float(I_cyl + 2.0 * (I_hemi_com + m_hemi * d ** 2))
