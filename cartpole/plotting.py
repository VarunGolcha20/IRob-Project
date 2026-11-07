"""Shared matplotlib style: readable labels/legends for the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
WEIGHT_DIR = ROOT / "weights"

# One fixed colour per model across every figure.
MODEL_COLORS = {
    "mujoco": "#222222",
    "analytical": "#1b9e77",
    "thin_rod": "#66a61e",
    "mlp": "#d95f02",
    "delan": "#7570b3",
}
MODEL_LABELS = {
    "mujoco": "MuJoCo (ground truth)",
    "analytical": "Analytical EL",
    "thin_rod": "Analytical (thin rod)",
    "mlp": "MLP (black box)",
    "delan": "DeLaN",
}


def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
