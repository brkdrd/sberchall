"""Are the low-scoring instances hard, or just badly searched?

`src/library.py` left one question standing, and it decides what is worth doing next. Best
P(ground) correlates with the spectral gap at **+0.587**, and by quartile of `E1 - E0`:

    narrowest  0.192   second  0.268   third  0.323   widest  0.505

Two readings fit that equally well.

- **Physics.** To put amplitude on the ground state alone, the circuit has to separate it
  from its nearest competitor. With `E1 - E0 ~ 0.03` and five layers there may be no angles
  that do it, and the quartile means are a ceiling.
- **Search.** A narrow gap makes the winning basin *small* in angle space, and possibly
  centred at a different gamma scale. 16k uniform starts would then miss it, and the
  quartile means are an artefact of how we look.

They are not distinguishable by staring at the correlation, and they imply opposite next
moves — one says package what we have, the other says spend inference budget where it pays.
So: take the worst instances, give each ~40x the budget across a gamma ladder, and see
whether anything moves. Seeded with each instance's best known angles, so the answer can
only be "no better" or "better", never an artefact of having lost ground.

This matters because of what a 0.91 leaderboard mean requires: with three quartiles at a
perfect 1.0 the mean is still 0.798. The narrow-gap instances have to move, or 0.91 is not
reachable by search at all and the number means something else.

Usage:
    python -m src.hard --npz runs/gscan.npz --instances 48
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .gscan import rung
from .qaoa_ref import QAOA, P as DEPTH
from .refine import refine, score

N_ANGLES = 2 * DEPTH


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--npz", type=Path, default=Path("runs/gscan.npz"))
    ap.add_argument("--out", type=Path, default=Path("runs/hard.npz"))
    ap.add_argument("--instances", type=int, default=48, help="how many worst to retry")
    ap.add_argument("--starts", type=int, default=32768, help="screened per instance per rung")
    ap.add_argument("--top", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scales", default="0.25,0.64,2.0,8.0")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lr-gamma", type=float, default=0.05)
    ap.add_argument("--lr-beta", type=float, default=0.03)
    ap.add_argument("--seed-steps", type=int, default=600,
                    help="Adam steps on the instance's already-best angles, as a floor")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32, device=dev)
    npz = np.load(args.npz)
    prev_p = torch.tensor(npz["full_p"], dtype=torch.float32, device=dev)
    prev_a = torch.tensor(npz["full_angles"], dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    scales = [float(s) for s in args.scales.split(",")]

    sel = prev_p.argsort()[:args.instances]
    E = sim.energies(h).sort(dim=1).values
    gap = E[:, 1] - E[:, 0]

    budget = len(scales) * args.starts + len(scales) * args.top * args.steps * 2
    print(f"retrying the {len(sel)} worst instances of {h.shape[0]} at "
          f"~{budget:,} forward passes each")
    print(f"they currently average P {prev_p[sel].mean():.5f} "
          f"(all 500 average {prev_p.mean():.5f}); their gaps average "
          f"{gap[sel].mean():.4f} against {gap.mean():.4f} overall\n")

    # floor: polish what each instance already has, so "no improvement" is a real answer
    best_a, best_p = refine(sim, h[sel], prev_a[sel], args.seed_steps, args.lr_gamma,
                            args.lr_beta, args.chunk, p0=score(sim, h[sel], prev_a[sel]))
    print(f"after {args.seed_steps} more Adam steps on their existing angles: "
          f"mean P {best_p.mean():.5f}\n")
    print(f"{'|gamma|<=':>10} {'mean P':>9} {'median':>9} {'max':>9} {'improved':>9}")

    for gmax in scales:
        t0 = time.time()
        r = rung(sim, h, sel, args.starts, args.top, args.steps, args.lr, gmax, gen,
                 dev, args.chunk, init="uniform")
        better = r["best"] > best_p
        best_p = torch.where(better, r["best"], best_p)
        best_a = torch.where(better.unsqueeze(-1), r["angles"], best_a)
        print(f"{gmax:10.2f} {best_p.mean().item():9.5f} {best_p.median().item():9.5f} "
              f"{best_p.max().item():9.5f} {int(better.sum()):9d}   [{time.time() - t0:.0f}s]")

    lift = best_p - prev_p[sel]
    print(f"\nworst {len(sel)}: {prev_p[sel].mean():.5f} -> {best_p.mean():.5f} "
          f"({best_p.mean() / prev_p[sel].mean():.2f}x)")
    print(f"  instances improved by >0.01: {int((lift > 0.01).sum())} of {len(sel)}")
    print(f"  largest single lift: {lift.max():.5f}")
    full = prev_p.clone()
    full[sel] = best_p
    print(f"  all 500 would move {prev_p.mean():.5f} -> {full.mean():.5f}")
    print(f"\nand the answer this run exists to give:")
    if lift.mean() < 0.01:
        print("  the worst instances did not move under ~40x the budget across four gamma")
        print("  scales. Their scores are a property of the instance, not of the search, so")
        print("  a 0.91 mean is not reachable this way and the effort belongs elsewhere.")
    else:
        print("  the worst instances moved. The quartile table was an artefact of where we")
        print("  looked, and inference budget spent per-instance on narrow-gap cases pays.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, sel=sel.cpu().numpy(), before=prev_p[sel].cpu().numpy(),
             after=best_p.cpu().numpy(), angles=best_a.cpu().numpy(),
             gap=gap[sel].cpu().numpy())
    print(f"\nwrote {args.out}")
    return {"before": prev_p[sel].mean().item(), "after": best_p.mean().item()}


if __name__ == "__main__":
    main()
