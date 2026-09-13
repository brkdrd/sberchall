"""Do the angles that win on one instance win on another?

The gamma sweep left a distribution, not a number: over 500 instances at 16k starts each,
best P(ground) ran from 0.029 to **0.9956**, median 0.256, mean 0.312. Some instances reach
the top of the scale at p=5, so the p=5 ceiling is not 0.33 — the mean is held down by
instances where a random multistart lands in the wrong basin. Quadrupling the starts
(4096 -> 16384) moved the mean by nothing, which says the same thing from the other side:
this is not a budget problem, it is a *where to look* problem.

That points somewhere specific. `h` enters the cost as a linear term on a fixed `J`, so
neighbouring instances have neighbouring landscapes, and a set of angles that works on one
should work on its neighbours. If that is true then the 500 winning angle vectors already
found are a **library**, and the right move for an instance we do badly on is to try the
other 499 before searching again.

This module measures exactly that, and it is cheap — 500 instances x 500 library entries is
250k forward passes, under a second on the GPU:

- `best-of-library` per instance, against what that instance's own search found;
- a **greedy cover**: how many library entries are needed to get most of the gain. If ten
  vectors carry it, the good basins are shared and the task is classification, not search —
  which is also a far better fit for a 10-minute inference limit than any search is;
- the same after an Adam polish, since a transferred vector lands near a basin rather than
  in it;
- and whether what is left correlates with the spectral gap, which is the test of whether
  the stragglers are intrinsically hard or merely unlucky.

Usage:
    python -m src.library --npz runs/gscan.npz --out runs/library_submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .predict import write_submission
from .qaoa_ref import QAOA, P as DEPTH
from .refine import P_FLOOR, refine, score

N_ANGLES = 2 * DEPTH


def collect(npz):
    """Every angle vector the sweep saved, from the full pass and from each rung."""
    out = []
    for key in npz.files:
        if not key.startswith("angles_") and key not in ("full_angles", "deep_angles"):
            continue
        a = np.asarray(npz[key], dtype=np.float32)
        if a.ndim == 2 and a.shape[1] == N_ANGLES:
            out.append(a)
    lib = np.unique(np.concatenate(out, axis=0).round(6), axis=0)
    return torch.tensor(lib, dtype=torch.float32)


@torch.no_grad()
def cross_evaluate(sim, h, lib, chunk=64):
    """P(ground) for every (instance, library entry) pair -> (N, L)."""
    n = h.shape[0]
    cols = []
    for lo in range(0, lib.shape[0], chunk):
        block = lib[lo:lo + chunk]
        b = block.shape[0]
        ang = block.unsqueeze(0).expand(n, -1, -1).reshape(n * b, N_ANGLES)
        hrep = h.repeat_interleave(b, dim=0)
        cols.append(score(sim, hrep, ang, 8192).view(n, b))
    return torch.cat(cols, dim=1)


def greedy_cover(table, k):
    """Pick library entries one at a time, each time the one that most lifts the mean.

    The question this answers is how *concentrated* the good basins are. A curve that
    saturates after a handful of entries means every instance is served by one of a few
    angle vectors, and predicting which one is a classification problem.
    """
    n, _ = table.shape
    best = torch.full((n,), -1.0, device=table.device)
    picks, curve = [], []
    for _ in range(k):
        gain = torch.maximum(table, best.unsqueeze(1)).mean(dim=0)
        j = int(gain.argmax())
        picks.append(j)
        best = torch.maximum(table[:, j], best)
        curve.append(best.mean().item())
    return picks, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--npz", type=Path, default=Path("runs/gscan.npz"))
    ap.add_argument("--h", type=Path, default=None, help="default: h_train from --data-dir")
    ap.add_argument("--out", type=Path, default=Path("runs/library_submission.csv"))
    ap.add_argument("--top", type=int, default=8, help="library entries polished per instance")
    ap.add_argument("--polish", type=int, default=400)
    ap.add_argument("--lr-gamma", type=float, default=0.05)
    ap.add_argument("--lr-beta", type=float, default=0.03)
    ap.add_argument("--cover", type=int, default=24, help="length of the greedy cover curve")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    hp = args.h or (args.data_dir / "h_train.npy")
    h = torch.tensor(np.load(hp), dtype=torch.float32, device=dev)
    lib = collect(np.load(args.npz)).to(dev)
    n = h.shape[0]
    print(f"library: {lib.shape[0]} distinct angle vectors from {args.npz}")
    print(f"instances: {n} from {hp}\n")

    table = cross_evaluate(sim, h, lib)
    own = table.diagonal() if table.shape[1] == n else None
    best_lib, which = table.max(dim=1)
    print(f"best-of-library, no search at all: mean P {best_lib.mean().item():.5f}, "
          f"median {best_lib.median().item():.5f}, min {best_lib.min().item():.5f}")
    if own is not None:
        print(f"each instance's own searched angles:  mean P {own.mean().item():.5f}")
    print(f"distinct library entries actually chosen: {len(set(which.tolist()))}\n")

    picks, curve = greedy_cover(table, min(args.cover, lib.shape[0]))
    print("greedy cover — mean P using only the best k library entries:")
    for i, v in enumerate(curve, 1):
        if i <= 8 or i % 4 == 0:
            print(f"    k={i:3d}  mean P {v:.5f}")
    print()

    idx = table.topk(min(args.top, lib.shape[0]), dim=1).indices
    ang = lib[idx.reshape(-1)]
    hrep = h.repeat_interleave(idx.shape[1], dim=0)
    ang, p = refine(sim, hrep, ang, args.polish, args.lr_gamma, args.lr_beta, args.chunk)
    p = p.view(n, -1)
    j = p.argmax(dim=1)
    rows = torch.arange(n, device=dev)
    final_p = p[rows, j]
    final_ang = ang.view(n, -1, N_ANGLES)[rows, j]
    print(f"after {args.polish} Adam steps on the top {idx.shape[1]}: "
          f"mean P {final_p.mean().item():.5f}, median {final_p.median().item():.5f}, "
          f"min {final_p.min().item():.5f}, max {final_p.max().item():.5f}")

    E = sim.energies(h).sort(dim=1).values
    gap = (E[:, 1] - E[:, 0]).cpu()
    lp = final_p.clamp_min(P_FLOOR).log().cpu()
    r = np.corrcoef(gap.numpy(), lp.numpy())[0, 1]
    print(f"\ncorrelation of log P with the E1-E0 gap: {r:+.3f}")
    q = torch.quantile(gap, torch.tensor([0.25, 0.5, 0.75]))
    for lo, hi, lab in ((-1e9, q[0], "narrowest quartile"), (q[0], q[1], "second"),
                        (q[1], q[2], "third"), (q[2], 1e9, "widest quartile")):
        m = (gap > lo) & (gap <= hi)
        print(f"    gap {lab:20s} n={int(m.sum()):3d}  mean P {final_p.cpu()[m].mean():.5f}")

    write_submission(args.out, final_ang)
    return {"best_of_library": best_lib.mean().item(),
            "after_polish": final_p.mean().item(), "library_size": lib.shape[0]}


if __name__ == "__main__":
    main()
