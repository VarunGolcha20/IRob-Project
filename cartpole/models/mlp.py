"""Black-box baseline: an MLP mapping (x, sin th, cos th, xd, thd, tau_x) -> (xdd, thdd).

Inputs and outputs are standardised with training-set statistics, so the
network works with O(1) numbers. Same activation (softplus) as DeLaN, so
smoothness is not a confound. 3 x 128 hidden units: 34 178 parameters, matched
to DeLaN's 34 310.
"""
from __future__ import annotations

import torch
from torch import nn

from ..dataset import features


def mlp(n_in: int, n_out: int, hidden: int, depth: int) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Softplus()]
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class MLPModel(nn.Module):
    name = "mlp"

    def __init__(self, stats: dict, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.config = dict(hidden=hidden, depth=depth)
        self.net = mlp(6, 2, hidden, depth)
        for k in ("in_mean", "in_std", "qdd_mean", "qdd_std"):
            self.register_buffer(k, torch.as_tensor(stats[k]).clone().float())

    def qdd(self, q, qd, tau):
        z = (features(q, qd, tau) - self.in_mean) / self.in_std
        return self.net(z) * self.qdd_std + self.qdd_mean
