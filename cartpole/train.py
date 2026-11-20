"""Training loop shared by the MLP and DeLaN (PLAN 3.3-3.4).

Loss (identical for both models):
    L_fwd = mean_i sum_j ((qdd_hat_ij - qdd_ij) / sigma_j)^2,   sigma_j = train std of axis j
DeLaN may add lam_inv * L_inv with
    L_inv = ((tau_hat_x - tau_x) / sigma_tau)^2 + (tau_hat_th / (sg(M22) sigma_thdd))^2
The theta row is divided by the model's own detached M22 (sg = stop-gradient),
so it cannot be reduced by shrinking the theta-row terms (M22, V, b_th).

Fixed budget of gradient steps (equal compute for every N), Adam with cosine
decay, gradient clipping, validation every `eval_every` steps, best
checkpoint kept, early stop after `patience` evaluations without improvement.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from .dataset import build
from .models import DeLaN, MLPModel, make_model
from .plotting import WEIGHT_DIR

DEFAULTS = dict(steps=30_000, batch=512, lr=1e-3, lr_min=1e-5, lam_inv=0.0,
                eval_every=500, patience=10, clip=10.0, threads=2)


def run_name(cfg: dict) -> str:
    name = f"{cfg['kind']}_{cfg['regime']}{cfg['N']}_s{cfg['seed']}"
    if cfg.get("lam_inv", 0.0) > 0:
        name += f"_inv{cfg['lam_inv']:g}"
    return name


def forward_loss(model, batch, stats):
    pred = model.qdd(batch["q"], batch["qd"], batch["tau"])
    return (((pred - batch["qdd"]) / stats["qdd_std"]) ** 2).sum(-1).mean()


def inverse_loss(model: DeLaN, batch, stats):
    tau_hat, s = model.inverse(batch["q"], batch["qd"], batch["qdd"])
    r_x = (tau_hat[..., 0] - batch["tau"][..., 0]) / stats["tau_std"]
    r_th = tau_hat[..., 1] / (s["M22"].detach() * stats["qdd_std"][1])
    return (r_x ** 2 + r_th ** 2).mean()


@torch.no_grad()
def evaluate(model, data, stats, chunk: int = 8192) -> float:
    n = len(data["q"])
    tot = 0.0
    for i in range(0, n, chunk):
        b = {k: v[i:i + chunk] for k, v in data.items()}
        tot += forward_loss(model, b, stats).item() * len(b["q"])
    return tot / n


def train_one(cfg: dict, out_dir: Path = WEIGHT_DIR, verbose: bool = False) -> dict:
    cfg = {**DEFAULTS, **cfg}
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    data = build(cfg["regime"], cfg["N"], cfg["seed"])
    tr, va, stats = data["train"], data["val"], data["stats"]
    model = make_model(cfg["kind"], stats)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"], eta_min=cfg["lr_min"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    n = len(tr["q"])

    hist = dict(step=[], train=[], val=[])
    best = dict(val=float("inf"), step=0, state=None)
    run_loss, run_n, bad = 0.0, 0, 0
    t0 = time.time()
    step = 0
    for step in range(1, cfg["steps"] + 1):
        idx = torch.randint(0, n, (cfg["batch"],), generator=gen)
        batch = {k: v[idx] for k, v in tr.items()}
        loss = forward_loss(model, batch, stats)
        if cfg["lam_inv"] > 0 and isinstance(model, DeLaN):
            loss = loss + cfg["lam_inv"] * inverse_loss(model, batch, stats)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        run_loss += loss.item()
        run_n += 1
        if step % cfg["eval_every"] == 0:
            v = evaluate(model, va, stats)
            hist["step"].append(step)
            hist["train"].append(run_loss / run_n)
            hist["val"].append(v)
            run_loss, run_n = 0.0, 0
            if v < best["val"]:
                best = dict(val=v, step=step, state=copy.deepcopy(model.state_dict()))
                bad = 0
            else:
                bad += 1
            if verbose:
                print(f"{run_name(cfg)} step {step} train {hist['train'][-1]:.3e} val {v:.3e}", flush=True)
            if bad >= cfg["patience"]:
                break

    out = dict(
        name=run_name(cfg), cfg=cfg, kind=cfg["kind"], model_config=model.config,
        stats={k: v.clone() for k, v in stats.items()}, state_dict=best["state"],
        best_val=best["val"], best_step=best["step"], stopped_step=step,
        history=hist, chosen=data["chosen"], n_train=data["n_train"], n_val=data["n_val"],
        time_s=time.time() - t0, n_params=sum(p.numel() for p in model.parameters()),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_dir / f"{out['name']}.pt")
    return {k: v for k, v in out.items() if k not in ("state_dict", "stats", "history")}


def load_model(path, dtype=torch.float64):
    """Load a trained model (best checkpoint) for evaluation."""
    ck = torch.load(path, weights_only=False)
    if ck["kind"] == "mlp":
        model = MLPModel(ck["stats"], **ck["model_config"])
    else:
        model = DeLaN(**ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck
