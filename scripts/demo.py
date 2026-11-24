"""Phase 6 pre-evaluation demo: one command per part.

  python scripts/demo.py --part rollout --model delan      # learned rollout vs MuJoCo, animated
  python scripts/demo.py --part swingup --model delan      # MPPI swing-up in dm_control

--part rollout
    Replays the recorded actions of one test episode through the chosen model (RK4, float64) and
    animates it next to the MuJoCo trajectory (dark), with theta(t) and x(t) below. Starts at
    --start seconds, runs --seconds. Pick the data with --pool {R_test,E_test,S_test} --episode N.
--part swingup
    Closed-loop MPPI swing-up (frozen config from results/p5_mppi_config.json, else defaults).
    --viewer opens the interactive dm_control viewer; --video out.mp4 renders a video;
    with neither, it runs headless and prints the score.

--model analytical | mlp | delan   (learned models: best-validation seed of regime --regime, N = 20)
--save PATH   (rollout) write a .gif / .mp4 instead of opening a window.
If no window can open (headless machine) the rollout is saved to figures/demo_rollout.gif.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole.data import load_episodes  # noqa: E402
from cartpole.integrators import rollout, wrap_angle  # noqa: E402
from cartpole.loading import best_run, get_params, load_any  # noqa: E402
from cartpole.mppi import MPPIConfig  # noqa: E402
from cartpole.plotting import DATA_DIR, FIG_DIR, MODEL_COLORS, MODEL_LABELS, RES_DIR  # noqa: E402

torch.set_default_dtype(torch.float64)


def resolve_model(args) -> str:
    if args.model == "analytical":
        return "analytical"
    return best_run(args.model, args.regime, 20)["path"]


def load_config() -> dict:
    path = RES_DIR / "p5_mppi_config.json"
    if path.exists():
        return json.loads(path.read_text())["config"]
    print("note: results/p5_mppi_config.json not found, using default MPPI settings")
    return MPPIConfig().to_dict()


# --------------------------------------------------------------------------------------
def part_rollout(args):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    p = get_params()
    model = load_any(resolve_model(args))
    eps, _ = load_episodes(DATA_DIR / f"{args.pool}.npz")
    e = eps[args.episode]
    T, s = int(round(args.seconds / p.dt)), int(round(args.start / p.dt))
    if s + T > len(e["u"]):
        sys.exit(f"window {args.start}-{args.start + args.seconds}s exceeds the {len(e['u']) * p.dt:.0f}s episode")
    if e["limit_active"][s:s + T].any():
        print("note: the rail limit is active inside this window; no model includes it (ISSUES A2)")
    QT = np.concatenate([e["qpos"], e["qpos_final"][None]])[s:s + T + 1]
    QDT = np.concatenate([e["qvel"], e["qvel_final"][None]])[s:s + T + 1]
    tau = torch.as_tensor(np.stack([p.gear * e["u"][s:s + T], np.zeros(T)], 1))
    with torch.no_grad():
        Q, _ = rollout(model, torch.as_tensor(QT[0]), torch.as_tensor(QDT[0]), tau, p.dt)
    Q = Q.numpy()
    t = np.arange(T + 1) * p.dt
    th_true, th_hat = wrap_angle(QT[:, 1]), wrap_angle(Q[:, 1])
    err = np.abs(wrap_angle(Q[:, 1] - QT[:, 1]))
    valid = t[np.argmax(err > 0.1)] if (err > 0.1).any() else t[-1]
    color = MODEL_COLORS[args.model]
    name = MODEL_LABELS[args.model]

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    xs = np.concatenate([QT[:, 0], Q[:, 0]])
    ax0.set(xlim=(xs.min() - 1.2, xs.max() + 1.2), ylim=(-1.2, 1.3), aspect="equal",
            title=f"{name} vs MuJoCo ({args.pool}, episode {args.episode}, t0 = {args.start:g} s); "
                  f"first |error| > 0.1 rad at {valid:.2f} s")
    ax0.axhline(0, color="k", lw=0.6)
    ax1.plot(t, th_true, color=MODEL_COLORS["mujoco"], label="MuJoCo")
    ax1.plot(t, th_hat, color=color, ls="--", label=name)
    ax1.set(xlabel="t [s]", ylabel=r"wrapped $\theta$ [rad]")
    ax1.legend(fontsize=9)
    ax2.plot(t, QT[:, 0], color=MODEL_COLORS["mujoco"])
    ax2.plot(t, Q[:, 0], color=color, ls="--")
    ax2.set(xlabel="t [s]", ylabel="cart x [m]")
    cur1, cur2 = ax1.axvline(0, color="k", lw=0.8), ax2.axvline(0, color="k", lw=0.8)

    L = 2 * p.l
    artists = []
    for traj, col, ls in ((QT, MODEL_COLORS["mujoco"], "-"), (Q, color, "--")):
        cart = patches.Rectangle((0, -0.08), 0.3, 0.16, fc="none", ec=col, lw=1.8, ls=ls)
        ax0.add_patch(cart)
        (pole,) = ax0.plot([], [], color=col, lw=3, ls=ls, solid_capstyle="round")
        artists.append((traj, cart, pole))

    def draw(i):
        out = []
        for traj, cart, pole in artists:
            x, th = traj[i, 0], traj[i, 1]
            cart.set_xy((x - 0.15, -0.08))
            pole.set_data([x, x + L * np.sin(th)], [0, L * np.cos(th)])
            out += [cart, pole]
        cur1.set_xdata([t[i]])
        cur2.set_xdata([t[i]])
        return out + [cur1, cur2]

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, draw, frames=range(0, T + 1, 2), interval=20, blit=False)
    save = args.save
    if save is None:
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            save = str(FIG_DIR / "demo_rollout.gif")
            print(f"no window available, saving to {save}")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        if save.endswith(".mp4"):
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            writer = animation.FFMpegWriter(fps=50)
        else:
            writer = animation.PillowWriter(fps=50)
        ani.save(save, writer=writer)
        print("saved", save)
    else:
        plt.show()


# --------------------------------------------------------------------------------------
def part_swingup(args):
    from cartpole.closedloop import run_swingup
    from cartpole.mppi import MPPIController
    from cartpole.params import make_env

    p = get_params()
    model = load_any(resolve_model(args))
    cfg = MPPIConfig.from_dict(load_config())

    if args.viewer:
        from dm_control import viewer
        env = make_env(args.seed)
        ctrl = MPPIController(model, p, cfg, args.seed)

        def policy(time_step):
            if time_step.first():
                ctrl.reset()
            d = env.physics.data
            return np.array([ctrl(d.qpos.copy(), d.qvel.copy())])

        print(f"viewer: MPPI with {args.model}, seed {args.seed} (close the window to stop)")
        viewer.launch(env, policy=policy)
        return

    frames = []
    hook = None
    if args.video:
        state = {"k": 0}

        def hook(physics, t, u):
            if state["k"] % 4 == 0:                       # 25 fps
                frames.append(physics.render(height=360, width=480, camera_id=0))
            state["k"] += 1

    print(f"running MPPI with {args.model} on seed {args.seed} (10 s, expect about a minute) ...", flush=True)
    res = run_swingup(model, cfg, args.seed, hook=hook)
    sc = res["score"]
    print(json.dumps(sc, indent=2))
    print("SUCCESS" if sc["success"] else "FAILED",
          f"(time to balance {sc['time_to_balance']}, max|x| {sc['max_abs_x']:.2f} m, rail hit {sc['limit_hit']})")
    if args.video:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.video, fps=25) as w:
            for f in frames:
                w.append_data(f)
        print("saved", args.video)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["rollout", "swingup"])
    ap.add_argument("--model", default="delan", choices=["analytical", "mlp", "delan"])
    ap.add_argument("--regime", default="B", choices=["A", "B"], help="training regime of the learned model (N = 20)")
    ap.add_argument("--seed", type=int, default=0, help="swingup: env seed")
    ap.add_argument("--pool", default="E_test", choices=["R_test", "E_test", "S_test"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=float, default=2.0, help="rollout window start [s]")
    ap.add_argument("--seconds", type=float, default=2.0, help="rollout window length [s]")
    ap.add_argument("--save", default=None, help="rollout: write .gif/.mp4 instead of opening a window")
    ap.add_argument("--viewer", action="store_true", help="swingup: interactive dm_control viewer")
    ap.add_argument("--video", default=None, help="swingup: write an .mp4")
    args = ap.parse_args()
    torch.set_num_threads(4)
    (part_rollout if args.part == "rollout" else part_swingup)(args)


if __name__ == "__main__":
    main()
