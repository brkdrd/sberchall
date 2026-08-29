"""Learn *where to start* the classical optimiser, not how to replace it.

Adam through the differentiable simulator is already the best angle-finder here; its only
real failure mode is converging into a bad basin. `src/basins.py` labels, for each
instance, the random starts that did reach the best-known optimum. This module trains a
model `h -> K candidate starts` against those labels, and inference is unchanged
otherwise: propose K starts, run the same Adam polish from each, keep the best.

Why the loss is a *minimum* over candidates. For a fixed `h` the good starts form several
disjoint blobs (one per basin). An L2 regression to all of them fits their mean, which
sits in no basin at all — the failure that caps a plain regression baseline. Taking, for
each prediction, the distance to its *nearest* label lets different heads own different
blobs, so nothing is averaged across modes.

Two details that matter more than the architecture:

- **beta distance is circular.** `beta` is pi-periodic, i.e. period 2 in normalised
  units, so a prediction at 0.99 and a label at -0.99 are 0.02 apart, not 1.98.
- **the loss needs both directions.** Nearest-label-per-prediction alone permits all K
  heads to collapse onto one blob; the reverse term (nearest-prediction-per-label) is
  what forces them to cover the basins. Together they are a Chamfer distance.

Usage:
    python -m src.proposer --basins runs/basins.npz --out-dir runs/proposer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .angles import DEPTH, angle_scale, canonicalise, to_angles
from .model import COND_DIM
from .predict import polish
from .qaoa_ref import QAOA
from .schedules import sample_starts

N_ANGLES = 2 * DEPTH
GAMMA_BOX = 3.0        # tanh bound on gamma in normalised units; beta needs only one period


class Proposer(nn.Module):
    """h -> K starting points. A shared trunk plus K learned query embeddings.

    The queries are what break the symmetry between heads: without them a single
    `Linear(d, K * 10)` also works, but the heads share no structure and specialise more
    slowly. Outputs are squashed into a sane box — one full period for beta, a few units
    for gamma — so an untrained model proposes valid starts rather than divergent ones.
    """

    def __init__(self, k=10, d=256, n_layers=3):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(nn.Linear(COND_DIM, d), nn.SiLU())
        self.body = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
            for _ in range(n_layers)
        )
        self.query = nn.Parameter(torch.randn(k, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(),
                                  nn.Linear(d, N_ANGLES))

    def forward(self, h):
        x = self.trunk(h)
        for blk in self.body:
            x = x + blk(x)
        x = x.unsqueeze(1) + self.query.unsqueeze(0)          # (B, K, d)
        out = self.head(x)
        g = GAMMA_BOX * torch.tanh(out[..., :DEPTH])
        b = torch.tanh(out[..., DEPTH:])
        return torch.cat([g, b], dim=-1)                       # (B, K, 10)


def pairwise_sq(pred, tgt):
    """Squared distance (B, K, M) with beta measured circularly (period 2)."""
    d = pred.unsqueeze(2) - tgt.unsqueeze(1)                   # (B, K, M, 10)
    g, b = d[..., :DEPTH], d[..., DEPTH:]
    b = (b + 1.0) % 2.0 - 1.0                                  # wrap into [-1, 1)
    return (g ** 2).sum(-1) + (b ** 2).sum(-1)


def chamfer(pred, tgt, mask, coverage=1.0):
    """Chamfer distance between K predictions and a padded, masked label set.

    precision: each prediction is pulled to its nearest label   -> predictions stay valid
    coverage : each label pulls its nearest prediction          -> predictions stay diverse
    """
    d = pairwise_sq(pred, tgt)                                 # (B, K, M)
    big = torch.finfo(d.dtype).max
    dm = d.masked_fill(~mask.unsqueeze(1), big)
    precision = dm.min(dim=2).values.mean(dim=1)               # (B,)
    cov = d.masked_fill(~mask.unsqueeze(1), big).min(dim=1).values
    cov = (cov * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return (precision + coverage * cov).mean(), precision.mean(), cov.mean()


def load_basins(path, max_per_instance=32, per_basin=8):
    """Padded label tensors from a basins.npz, balanced across basins.

    Capping per basin matters: a wide basin contributes far more sampled starts than a
    narrow one, and without the cap the loss would simply follow that volume ratio
    instead of covering every basin the instance has.
    """
    z = np.load(path)
    h, owner, start, basin = z["h"], z["owner"], z["start"], z["basin"]
    n = len(h)
    keep = [[] for _ in range(n)]
    order = np.argsort(owner, kind="stable")
    for j in order:
        i = owner[j]
        if len(keep[i]) >= max_per_instance:
            continue
        if sum(1 for x in keep[i] if basin[x] == basin[j]) < per_basin:
            keep[i].append(j)
    have = [i for i in range(n) if keep[i]]
    m = max(len(keep[i]) for i in have)
    tgt = np.zeros((len(have), m, N_ANGLES), dtype=np.float32)
    mask = np.zeros((len(have), m), dtype=bool)
    for r, i in enumerate(have):
        tgt[r, :len(keep[i])] = start[keep[i]]
        mask[r, :len(keep[i])] = True
    print(f"{path}: {len(have)}/{n} instances usable, "
          f"{mask.sum() / len(have):.1f} labels each (max {m}), "
          f"{len(np.unique(basin))} basin ids seen")
    return (torch.tensor(h[have]), torch.tensor(tgt), torch.tensor(mask),
            torch.tensor(z["scale"]))


@torch.no_grad()
def polish_from(sim, h, u0, scale, steps, lr, chunk):
    """Adam from each of K starts, keep the best endpoint per instance.

    Selection happens *after* polishing, for the same reason it does in `src.validate`:
    the best start is not necessarily the one whose basin has the best floor.
    """
    n = h.shape[0]
    best_u = torch.zeros(n, N_ANGLES, device=h.device)
    best_p = torch.full((n,), -1.0, device=h.device)
    for j in range(u0.shape[0]):
        with torch.enable_grad():
            u = polish(sim, h, u0[j].detach(), scale, steps, lr=lr, chunk=chunk)
        a = to_angles(u, scale)
        p = sim.p_ground(h, a[:, :DEPTH], a[:, DEPTH:])
        m = p > best_p
        best_p = torch.where(m, p, best_p)
        best_u = torch.where(m.unsqueeze(1), u, best_u)
    return best_u, best_p


@torch.no_grad()
def random_starts(k, n, device, seed, init="tqa"):
    """The non-learned arm of the comparison.

    This must use the *same* init family the labels came from, or the head-to-head is a
    strawman: beating uniform-box starts is easy and proves nothing once we know the box
    is the wrong place to sample from.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return sample_starts(k * n, init, device, g).view(k, n, N_ANGLES)


@torch.no_grad()
def head_to_head(model, sim, h, scale, steps, lr, chunk, seed=0, init="tqa"):
    """The only evaluation that matters: learned starts vs the same number of random ones.

    Both arms get an identical Adam budget, so any difference is the starts alone.
    """
    model.eval()
    k, n = model.k, h.shape[0]
    _, pl = polish_from(sim, h, model(h).transpose(0, 1), scale, steps, lr, chunk)
    _, pr = polish_from(sim, h, random_starts(k, n, h.device, seed, init), scale, steps,
                        lr, chunk)
    model.train()
    return {"learned": pl.mean().item(), "random": pr.mean().item()}


def load_model(ckpt, device):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = c["config"]
    m = Proposer(k=cfg["k"], d=cfg["d_model"], n_layers=cfg["layers"]).to(device)
    m.load_state_dict(c["model"])
    m.eval()
    return m, c["angle_scale"].to(device), c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basins", type=Path, default=Path("runs/basins.npz"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/proposer"))
    ap.add_argument("--k", type=int, default=10, help="starts proposed per instance")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coverage", type=float, default=1.0, help="weight of the diversity term")
    ap.add_argument("--max-per-instance", type=int, default=32)
    ap.add_argument("--per-basin", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-instances", type=int, default=64)
    ap.add_argument("--polish-steps", type=int, default=150)
    ap.add_argument("--polish-lr", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--init", default="tqa", choices=("uniform", "tqa", "ramp"),
                    help="init family for the non-learned baseline arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h, tgt, mask, scale = load_basins(args.basins, args.max_per_instance, args.per_basin)
    h, tgt, mask, scale = h.to(dev), tgt.to(dev), mask.to(dev), scale.to(dev)

    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h_eval = torch.tensor(np.load(args.data_dir / "h_train.npy"),
                          dtype=torch.float32, device=dev)[:args.eval_instances]

    model = Proposer(k=args.k, d=args.d_model, n_layers=args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    best_metric, t0 = 0.0, time.time()
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, h.shape[0], (min(args.batch, h.shape[0]),), device=dev)
        loss, prec, cov = chamfer(model(h[idx]), tgt[idx], mask[idx], args.coverage)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 100 == 0 or it == 1:
            print(f"iter {it:5d}  loss {loss.item():7.4f}  "
                  f"precision {prec.item():7.4f}  coverage {cov.item():7.4f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)

        if it % args.eval_every == 0 or it == args.iters:
            r = head_to_head(model, sim, h_eval, scale, args.polish_steps, args.polish_lr,
                             args.chunk, seed=args.seed, init=args.init)
            lift = r["learned"] / max(r["random"], 1e-12)
            print(f"eval  iter {it}: best-of-{args.k} P — learned {r['learned']:.4f}  "
                  f"vs random {r['random']:.4f}   ({lift:.3f}x)", flush=True)
            ck = {"model": model.state_dict(), "config": cfg, "metric": r["learned"],
                  "random_baseline": r["random"], "iter": it, "angle_scale": scale.cpu()}
            torch.save(ck, args.out_dir / "last.pt")
            if r["learned"] > best_metric:
                best_metric = r["learned"]
                torch.save(ck, args.out_dir / "best.pt")
                print(f"      new best -> {args.out_dir / 'best.pt'}", flush=True)

    print(f"done. best learned best-of-{args.k} P: {best_metric:.4f}")
    return {"best_ckpt": args.out_dir / "best.pt", "best_metric": best_metric}


if __name__ == "__main__":
    main()
