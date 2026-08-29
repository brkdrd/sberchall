"""Basin sampling: which starting points flow to the optimum?

The classical optimiser is the best angle-finder we have; its only real failure is landing
in a bad basin. So instead of learning to replace it, learn where to *start* it.

You cannot get that by running gradient descent backwards from a known optimum — the
reverse-time flow is expanding (the basin boundary is a repeller, so integration error
blows up exponentially) and Adam, with momentum and per-coordinate scaling, is not a
reversible map at all. The stable way to get the same information is to run the optimiser
*forwards* from many random starts and keep the start alongside the endpoint it reached.
Every start that converged to the instance's best-known optimum is a labelled point in
that optimum's basin of attraction, which is exactly the supervision we want.

Two structural facts make the labels far cheaper than they look (see `angles.canonicalise`):

- the flow commutes with the 64-element symmetry group, so a start may be folded into the
  fundamental domain before it is run and it still converges to an equivalent optimum.
  That shrinks the region the proposer must cover by 64x;
- folding the *endpoints* is what makes "did these two starts reach the same optimum?" a
  well-posed question, so basins can be identified exactly rather than by clustering
  heuristics.

Usage:
    python -m src.basins --instances 2000 --starts 256 --out runs/basins.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .angles import DEPTH, angle_scale, canonicalise, to_angles
from .predict import polish
from .qaoa_ref import QAOA
from .schedules import sample_starts


def p_of(sim, h, u, scale):
    a = to_angles(u, scale)
    with torch.no_grad():
        return sim.p_ground(h, a[:, :DEPTH], a[:, DEPTH:])


def run_starts(sim, h, scale, n_starts, steps, lr, chunk, generator, init="tqa"):
    """Adam from `n_starts` canonical random starts per instance.

    Starts are batched together rather than run one at a time: every start is an
    independent Adam trajectory, so packing as many as fit into `chunk` rows turns
    `n_starts` sequential kernel launches into `n_starts * N / chunk` of them. On a GPU
    that is the difference between minutes and hours.

    Returns (starts, ends, p) each shaped (n_starts, N, ...) — starts and endpoints in
    normalised units, p the final P(ground).
    """
    n = h.shape[0]
    per = max(1, chunk // n)                       # starts per batch, so rows <= chunk
    starts, ends, ps = [], [], []
    for lo in range(0, n_starts, per):
        s = min(per, n_starts - lo)
        u0 = sample_starts(s * n, init, h.device, generator)
        hr = h.repeat(s, 1)                        # [h; h; ...] -> row = start * N + inst
        u1 = polish(sim, hr, u0, scale, steps, lr=lr, chunk=chunk)
        starts.append(u0.view(s, n, -1))
        ends.append(canonicalise(u1).view(s, n, -1))
        ps.append(p_of(sim, hr, u1, scale).view(s, n))
    return torch.cat(starts), torch.cat(ends), torch.cat(ps)


def label(starts, ends, ps, rel_tol, basin_eps):
    """Keep the starts that reached the instance's best-known optimum; id their basins.

    A start is 'good' when its final P is within `rel_tol` of the best P any start found
    for that instance. Two good starts share a basin when their canonical endpoints
    coincide to `basin_eps` — an exact test, because canonicalisation has already removed
    the 64 symmetry copies that would otherwise make identical optima look distinct.
    """
    n_starts, n = ps.shape
    best = ps.max(dim=0).values                          # (N,)
    good = ps >= best.unsqueeze(0) * (1.0 - rel_tol)     # (n_starts, N)

    owner, u0, pend, basin = [], [], [], []
    for i in range(n):
        idx = torch.nonzero(good[:, i], as_tuple=True)[0]
        e = ends[idx, i]                                 # (G_i, 10)
        # greedy exact clustering on canonical endpoints
        ids = torch.full((len(idx),), -1, dtype=torch.long)
        reps = []
        for j in range(len(idx)):
            hit = -1
            for r, rep in enumerate(reps):
                if torch.norm(e[j] - rep) < basin_eps:
                    hit = r
                    break
            if hit < 0:
                reps.append(e[j])
                hit = len(reps) - 1
            ids[j] = hit
        owner.append(torch.full((len(idx),), i, dtype=torch.long))
        u0.append(starts[idx, i])
        pend.append(ps[idx, i])
        basin.append(ids)
    return (torch.cat(owner), torch.cat(u0).cpu(), torch.cat(pend).cpu(), torch.cat(basin),
            best.cpu())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("runs/basins.npz"))
    ap.add_argument("--instances", type=int, default=2000, help="synthetic h ~ U(-1,1)")
    ap.add_argument("--starts", type=int, default=256, help="random starts per instance")
    ap.add_argument("--steps", type=int, default=150, help="Adam steps per start")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--rel-tol", type=float, default=0.02,
                    help="a start is 'good' if final P >= (1 - rel_tol) * best P")
    ap.add_argument("--basin-eps", type=float, default=0.05,
                    help="canonical endpoints closer than this are the same optimum")
    ap.add_argument("--block", type=int, default=250, help="instances processed at once")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock budget; 0 = unlimited. On expiry the labels gathered "
                         "so far are written, so the stage is always safe to cut short.")
    ap.add_argument("--chunk", type=int, default=4096, help="rows per Adam chunk")
    ap.add_argument("--init", default="tqa", choices=("uniform", "tqa", "ramp"),
                    help="where starts come from; 'uniform' is the old 10-D box, which "
                         "misses the smooth-schedule surface the optima lie on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h_ref = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                         device=dev)
    scale = angle_scale(sim, h_ref, device=dev)

    print(f"generating {args.instances} instances x {args.starts} starts x {args.steps} "
          f"Adam steps on {dev}  (init={args.init})", flush=True)
    out = {k: [] for k in ("h", "best_p", "start", "owner", "basin", "p_end")}
    t0, done = time.time(), 0
    budget_s = args.max_hours * 3600 if args.max_hours else None
    for lo in range(0, args.instances, args.block):
        n = min(args.block, args.instances - lo)
        # training instances are synthesised: h_train is i.i.d. U(-1, 1) and held out
        h = torch.rand(n, h_ref.shape[1], device=dev, generator=gen) * 2 - 1
        starts, ends, ps = run_starts(sim, h, scale, args.starts, args.steps, args.lr,
                                      args.chunk, gen, init=args.init)
        owner, u0, p_end, basin, best = label(starts, ends, ps, args.rel_tol,
                                              args.basin_eps)
        out["h"].append(h.cpu().numpy())
        out["best_p"].append(best.numpy())
        out["start"].append(u0.numpy())
        out["owner"].append((owner + done).numpy())
        out["basin"].append(basin.numpy())
        out["p_end"].append(p_end.numpy())
        done += n
        el = time.time() - t0
        print(f"  {done}/{args.instances} instances  ({el:.0f}s, "
              f"{el / done * args.instances / 60:.1f} min projected)  "
              f"mean best P {best.mean():.4f}  "
              f"good starts/instance {len(owner) / n:.1f}", flush=True)
        if budget_s and el > budget_s:
            print(f"  wall-clock budget of {args.max_hours:g} h reached -- keeping the "
                  f"{done} instances gathered so far", flush=True)
            break

    d = {k: np.concatenate(v) for k, v in out.items()}
    d["scale"] = scale.cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)

    n_basins = np.array([len(np.unique(d["basin"][d["owner"] == i])) for i in range(done)])
    hit = len(d["start"]) / (done * args.starts)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"good starts: {len(d['start'])} of {done * args.starts} "
          f"({hit:.1%} of random starts reach the optimum)")
    print(f"distinct basins per instance: mean {n_basins.mean():.2f}  max {n_basins.max()}")
    print(f"mean best P(ground): {d['best_p'].mean():.4f}")
    print(f"\nexpected P from k random starts: "
          + "  ".join(f"k={k}: {1 - (1 - hit) ** k:.1%}" for k in (1, 10, 50, 256)))
    return d


if __name__ == "__main__":
    main()
