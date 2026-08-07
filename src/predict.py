"""Inference: best-of-K rollouts on a file of h vectors -> submission.csv.

Usage:
    python -m src.predict --h data/raw/h_test.npy
Optionally polish the best angles with a few Adam steps through the simulator:
    python -m src.predict --h data/raw/h_test.npy --polish 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import AngleTransformer
from .qaoa_ref import QAOA, P as DEPTH
from .rollout import best_of_rollouts, log_p


def polish(sim, h, angles, steps, lr=0.03):
    a = angles.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (-log_p(sim, h, a).sum()).backward()
        opt.step()
    return a.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=Path, default=Path("data/raw/h_test.npy"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/submission.csv"))
    ap.add_argument("--restarts", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--polish", type=int, default=0, help="Adam refinement steps (0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=device)
    h = torch.tensor(np.load(args.h), dtype=torch.float32, device=device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AngleTransformer(
        d=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
        max_len=cfg["steps"] + 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    angles, p = best_of_rollouts(model, sim, h, args.steps, args.restarts, chunk=args.chunk)
    print(f"best-of-{args.restarts} rollouts: mean P(ground) = {p.mean().item():.4f}")

    if args.polish > 0:
        angles = polish(sim, h, angles, args.polish)
        p = sim.p_ground(h, angles[:, :DEPTH], angles[:, DEPTH:])
        print(f"after {args.polish} polish steps: mean P(ground) = {p.mean().item():.4f}")

    a = angles.cpu().numpy()
    header = ["id"] + [f"gamma_{i}" for i in range(DEPTH)] + [f"beta_{i}" for i in range(DEPTH)]
    rows = np.column_stack([np.arange(len(a)), a])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.out, rows, delimiter=",", header=",".join(header), comments="",
        fmt=["%d"] + ["%.8f"] * 2 * DEPTH,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
