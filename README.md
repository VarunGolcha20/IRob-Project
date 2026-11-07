# Deep Lagrangian Networks for MPPI Swing-Up of the Cart-Pole

This project learns the forward dynamics of the dm_control `cartpole/swingup` with a Deep Lagrangian Network (DeLaN). It compares DeLaN with a black-box MLP and with the analytical Euler–Lagrange model, both on prediction and inside an MPPI swing-up controller. The plan is in [PLAN.md](PLAN.md), and the per-phase derivations are in [docs/](docs/).

## Setup (Python 3.12, CPU only)

```bash
python3 -m virtualenv ~/irob_proj
source ~/irob_proj/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export MUJOCO_GL=egl      # headless rendering; use osmesa if egl is unavailable
```

If ROS is sourced in the shell, run commands with `env -u PYTHONPATH ...`. Otherwise ROS's Python path leaks into the venv.

## Reproducing results

| phase | command | outputs |
|---|---|---|
| 1. Analytical model vs MuJoCo | `python scripts/p1_validate.py` | `results/p1_validation.json`, `figures/p1_residuals.*`, `figures/p1_rk4_vs_mujoco.*` |
| tests | `pytest -q tests/` | — |

## Layout

```
cartpole/   params (read from MuJoCo), analytical model, RK4, data collection, plotting
scripts/    one entry point per phase
docs/       math notes per phase (PART1_EL.md, ...)
tests/      structural / validation tests
data/ weights/ results/ figures/ report/ minutes/
```
