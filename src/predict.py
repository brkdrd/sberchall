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
from .rollout import best_of_rollouts, log_p, restart_candidates


def polish(sim, h, angles, steps, lr=0.03, chunk=2048):
    out = []
    for lo in range(0, h.shape[0], chunk):
        hi = min(lo + chunk, h.shape[0])
        a = angles[lo:hi].clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            (-log_p(sim, h[lo:hi], a).sum()).backward()
            opt.step()
        out.append(a.detach())
    return torch.cat(out)


def polish_then_select(sim, h, cand_angles, cand_p, steps, top_m, lr=0.03):
    """Polish the top-M candidates per instance, keep the post-polish best.

    The best pre-polish point is not necessarily in the best basin — Adam
    drives each candidate to its own basin's optimum, so the winner must be
    chosen after polishing, not before.

    cand_angles: (R, N, 10), cand_p: (R, N). Returns (angles (N, 10), p (N,)).
    """
    n_r, n, _ = cand_angles.shape
    m = min(top_m, n_r)
    idx = cand_p.topk(m, dim=0).indices                       # (m, N)
    sel = cand_angles.gather(0, idx.unsqueeze(-1).expand(m, n, 10))
    flat_a = polish(sim, h.repeat(m, 1), sel.reshape(m * n, 10), steps, lr=lr)
    p = sim.p_ground(h.repeat(m, 1), flat_a[:, :DEPTH], flat_a[:, DEPTH:]).reshape(m, n)
    best = p.argmax(dim=0)
    cols = torch.arange(n, device=h.device)
    return flat_a.reshape(m, n, 10)[best, cols], p[best, cols]


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
    ap.add_argument("--top-m", type=int, default=16, help="candidates polished per instance")
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

    if args.polish > 0:
        cand_a, cand_p = restart_candidates(
            model, sim, h, args.steps, args.restarts, chunk=args.chunk
        )
        print(
            f"best-of-{args.restarts} rollouts: "
            f"mean P(ground) = {cand_p.max(dim=0).values.mean().item():.4f}"
        )
        angles, p = polish_then_select(sim, h, cand_a, cand_p, args.polish, args.top_m)
        print(
            f"polish top-{args.top_m} x {args.polish} steps, select after: "
            f"mean P(ground) = {p.mean().item():.4f}"
        )
    else:
        angles, p = best_of_rollouts(model, sim, h, args.steps, args.restarts, chunk=args.chunk)
        print(f"best-of-{args.restarts} rollouts: mean P(ground) = {p.mean().item():.4f}")

    write_submission(args.out, angles)


def write_submission(path, angles):
    """Write angles (N, 10) to the competition csv format: id, gamma_0..4, beta_0..4."""
    a = angles.detach().cpu().numpy()
    header = ["id"] + [f"gamma_{i}" for i in range(DEPTH)] + [f"beta_{i}" for i in range(DEPTH)]
    rows = np.column_stack([np.arange(len(a)), a])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path, rows, delimiter=",", header=",".join(header), comments="",
        fmt=["%d"] + ["%.8f"] * 2 * DEPTH,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
