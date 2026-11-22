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
| 2. Datasets (R, E, S, U pools) | `python scripts/p2_collect.py` | `data/*.npz`, `results/p2_dataset.json`, `figures/p2_coverage.*` |
| 3. Training (needs torch) | `python scripts/p3_train.py` | `weights/*.pt`, `results/p3_training.json` |
| 3. Identification, learned structure, curves | `python scripts/p3_structure.py` | `results/p3_identification.json`, `figures/p3_*` |
| 4. Open-loop evaluation (~5 min) | `python scripts/p4_eval.py` (`--plots-only` to re-plot) | `results/p4_eval.json`, `figures/p4_*` |
| 5. MPPI tuning (analytical, seeds 100–109) | `python scripts/p5_mppi.py tune` | `results/p5_tuning.csv`, `results/p5_mppi_config.json` |
| 5. MPPI evaluation (6 models × 20 seeds; MLP ≈ 2.5 min/episode) | `python scripts/p5_mppi.py eval --workers 2` then `python scripts/p5_mppi.py plots` | `results/p5_runs/`, `results/p5_closed_loop.json`, `figures/p5_*` |
| RESULTS.md tables | `python scripts/results_tables.py [p3\|p4\|p5]` | markdown on stdout |
| known issues / risks | see [docs/ISSUES.md](docs/ISSUES.md) | — |
| tests | `pytest -q tests/` | — |

**Torch-free path.** Phases 3 (structure) to 5 need only numpy, scipy and matplotlib. [cartpole/npdyn.py](cartpole/npdyn.py) reads `weights/*.pt` without torch and reproduces the trained models in float64. `pytest -q tests/test_npdyn.py` checks it against the stored validation losses and the MuJoCo data. Training (`p3_train.py`) and data collection (`p1`, `p2`) still need torch / MuJoCo.

## Layout

```
cartpole/   params (read from MuJoCo), analytical model, RK4, data collection, plotting
scripts/    one entry point per phase
docs/       math notes per phase (PART1_EL.md, ...)
tests/      structural / validation tests
data/ weights/ results/ figures/ report/ minutes/
```
