"""What number is the leaderboard actually showing?

Our own evidence now says 0.91 is not reachable as P(ground) on these instances:

- five optimisers (Adam multistart, CMA-ES, TuRBO, L-BFGS, a learned node-chain policy),
- up to 745k forward passes per instance,
- gamma scales from 0.16 to 64,
- a 756-vector transfer library cross-evaluated against all 500 instances,

all land between 0.31 and 0.32. Best P correlates with the E1-E0 gap at +0.587, and the
48 narrowest-gap instances moved 0.0736 -> 0.0980 under 11x budget, i.e. +0.0023 on the
overall mean. With the other three quartiles at a *perfect* 1.0 the mean is still 0.798.

So before concluding anything about anyone's score, check the cheaper explanation: that the
number being reported is a different statistic of the same state. `P(ground)` is one
summary of a 4096-vector of probabilities and there are several neighbours of it that are
standard in the QAOA literature and would read far higher on identical angles — notably
the **approximation ratio**, for which 0.9 at p=5 is an ordinary result, while P(ground)
0.9 is not.

This prints every such statistic for a submission we already have. If one of them lands on
0.91 for angles that score 0.32 under the shipped `QAOA.p_ground`, that is the answer, and
it says our angles were never the problem.

Usage:
    python -m src.metrics --submission runs/library_submission.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .qaoa_ref import QAOA, P as DEPTH

N_ANGLES = 2 * DEPTH
TOLERANCES = (1e-9, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
TOPK = (1, 2, 4, 8, 16, 32, 64)


def load_submission(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    if a.shape[1] == N_ANGLES + 1:
        a = a[:, 1:]
    return torch.tensor(a, dtype=torch.float32)


@torch.no_grad()
def report(sim, h, ang, chunk=250):
    """Every plausible reading of 'how good is this state', on identical angles."""
    n = h.shape[0]
    acc = {}
    for lo in range(0, n, chunk):
        hh, aa = h[lo:lo + chunk], ang[lo:lo + chunk]
        E = sim.energies(hh)
        prob = sim.probs(hh, aa[:, :DEPTH], aa[:, DEPTH:])
        Es, order = E.sort(dim=1)
        ps = prob.gather(1, order)
        emin, emax = Es[:, :1], Es[:, -1:]
        mean_E = (prob * E).sum(dim=1, keepdim=True)

        for t in TOLERANCES:
            acc.setdefault(f"P(E <= Emin + {t:g})", []).append(
                (prob * (E <= emin + t).float()).sum(dim=1))
        for k in TOPK:
            acc.setdefault(f"P(state in {k} lowest)", []).append(ps[:, :k].sum(dim=1))
        acc.setdefault("approximation ratio (Emax-<E>)/(Emax-Emin)", []).append(
            ((emax - mean_E) / (emax - emin)).squeeze(1))
        acc.setdefault("energy ratio <E>/Emin", []).append((mean_E / emin).squeeze(1))
        acc.setdefault("P(E <= Emin + 1% of span)", []).append(
            (prob * (E <= emin + 0.01 * (emax - emin)).float()).sum(dim=1))
        acc.setdefault("P(E <= Emin + 5% of span)", []).append(
            (prob * (E <= emin + 0.05 * (emax - emin)).float()).sum(dim=1))
        acc.setdefault("P(E below the mean of the spectrum)", []).append(
            (prob * (E <= E.mean(dim=1, keepdim=True)).float()).sum(dim=1))
        acc.setdefault("max single-state probability", []).append(prob.max(dim=1).values)
    return {k: torch.cat(v) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--submission", type=Path, default=Path("runs/library_submission.csv"))
    ap.add_argument("--h", type=Path, default=None)
    ap.add_argument("--target", type=float, default=0.91104,
                    help="the leaderboard number to look for")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    hp = args.h or (args.data_dir / "h_train.npy")
    h = torch.tensor(np.load(hp), dtype=torch.float32, device=dev)
    ang = load_submission(args.submission).to(dev)
    if ang.shape[0] != h.shape[0]:
        raise SystemExit(f"{args.submission} has {ang.shape[0]} rows, {hp} has {h.shape[0]}")

    print(f"{args.submission} on {hp}: {h.shape[0]} instances\n")
    res = report(sim, h, ang)
    official = res[f"P(E <= Emin + {1e-9:g})"].mean().item()
    print(f"{'statistic':<42} {'mean':>9} {'median':>9}   vs {args.target:g}")
    for k, v in res.items():
        m = v.mean().item()
        flag = "  <== matches the leaderboard" if abs(m - args.target) < 0.02 else ""
        print(f"{k:<42} {m:9.5f} {v.median().item():9.5f}{flag}")
    print(f"\nthe shipped QAOA.p_ground uses tolerance 1e-9: {official:.5f}")
    close = [k for k, v in res.items() if abs(v.mean().item() - args.target) < 0.02]
    print("\n" + ("statistics matching the leaderboard on these same angles:\n  "
                  + "\n  ".join(close) if close else
                  "no statistic of these angles lands near the leaderboard number."))
    return {"official": official, "matches": close}


if __name__ == "__main__":
    main()
