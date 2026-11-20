"""Learned dynamics models. Both implement qdd(q, qd, tau) like AnalyticalModel."""
from .delan import DeLaN  # noqa: F401
from .mlp import MLPModel  # noqa: F401


def make_model(kind: str, stats: dict, **kw):
    if kind == "mlp":
        return MLPModel(stats, **kw)
    if kind == "delan":
        return DeLaN(**kw)
    raise ValueError(kind)
