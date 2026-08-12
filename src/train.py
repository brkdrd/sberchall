"""Train the learned-optimizer transformer.

Training data is synthesised on the fly: h_train is i.i.d. U(-1, 1)
(KS-verified), so fresh h ~ U(-1, 1) each iteration gives unlimited instances
with zero distribution shift. h_train itself is held out for evaluation.

Usage (defaults are the intended run):
    python -m src.train
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import AngleTransformer, COND_DIM
from .qaoa_ref import QAOA
from .rollout import best_of_rollouts, rollout


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/longer"))
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8, help="rollout length")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.10, help="initial exploration std")
    ap.add_argument("--noise-final", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-restarts", type=int, default=16)
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def evaluate(model, sim, h_eval, args):
    model.eval()
    _, p = best_of_rollouts(
        model, sim, h_eval, args.steps, args.eval_restarts, chunk=args.eval_chunk
    )
    model.train()
    return p.mean().item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    J = np.load(args.data_dir / "J.npy")
    h_eval = torch.tensor(
        np.load(args.data_dir / "h_train.npy"), dtype=torch.float32, device=device
    )
    sim = QAOA(J, device=device)
    model = AngleTransformer(
        d=args.d_model, n_heads=args.heads, n_layers=args.layers, max_len=args.steps + 1
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    # later steps should land in better basins — weight the loss accordingly
    w = torch.arange(1, args.steps + 1, dtype=torch.float32, device=device)
    w = w / w.sum()

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    best_metric = 0.0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        frac = (it - 1) / max(args.iters - 1, 1)
        noise = args.noise_final + 0.5 * (args.noise - args.noise_final) * (
            1 + math.cos(math.pi * frac)
        )
        h = (torch.rand(args.batch, COND_DIM, device=device) * 2 - 1)

        losses, _, traj_logp = rollout(model, sim, h, args.steps, noise_std=noise)
        loss = sum(wt * l.mean() for wt, l in zip(w, losses))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 50 == 0 or it == 1:
            p_last = traj_logp[:, -1].exp().mean().item()
            p_start = traj_logp[:, 0].exp().mean().item()
            print(
                f"iter {it:5d}  loss {loss.item():7.3f}  "
                f"P start {p_start:.4f} -> last {p_last:.4f}  "
                f"noise {noise:.3f}  {(time.time() - t0):6.0f}s",
                flush=True,
            )

        if it % args.eval_every == 0 or it == args.iters:
            metric = evaluate(model, sim, h_eval, args)
            print(
                f"eval  iter {it}: mean best-of-{args.eval_restarts} "
                f"P(ground) on h_train = {metric:.4f}",
                flush=True,
            )
            ckpt = {"model": model.state_dict(), "config": cfg, "metric": metric, "iter": it}
            torch.save(ckpt, args.out_dir / "last.pt")
            if metric > best_metric:
                best_metric = metric
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"      new best -> {args.out_dir / 'best.pt'}", flush=True)

    print(f"done. best eval metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
