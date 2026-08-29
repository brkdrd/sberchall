"""Structured starting points: sample schedules, not boxes.

`init_units` draws each of the 10 angles independently and uniformly. That treats the
angle vector as 10 free coordinates, and it is why a random start almost never lands in a
good basin: optimal QAOA angles are not scattered through the box, they lie close to a
smooth *annealing schedule* — gamma ramping up with layer index, beta ramping down —
which is a roughly 2-parameter family inside a 10-dimensional space. Sampling the box to
find that surface is sampling a volume to hit a sheet.

This is the standard result, not a guess: a depth-p QAOA circuit run with
`gamma_l = (l/p) * s`, `beta_l = (1 - l/p) * s` is a Trotterised quantum annealing
schedule (Sack & Serbyn, *Quantum* 5, 491 (2021)), and it lands in the right basin far
more often than a random point. Zhou et al. (PRX 10, 021067 (2020)) make the same
observation from the other direction: optimal schedules are smooth in `l`, so they are
compactly described by a handful of low-frequency coefficients (their INTERP / FOURIER
heuristics).

**The scale caveat that matters here.** TQA is derived for problems where the cost and
mixer terms are comparable. They are not: this spectrum spans ~39, so one phase
revolution is `gamma ~ 0.16` rad while beta's period is `pi`. A single `s` in *radians*
would therefore be simultaneously far too large for gamma and too small for beta. The
schedule is built in the normalised units of `angles.py`, where a unit step means the
same thing in both halves, and converted to radians only at the simulator boundary.

Diagnostic — no optimisation at all, just the schedule:
    python -m src.schedules            # scan s, report mean P(ground) on h_train
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .angles import DEPTH, angle_scale, canonicalise, to_angles
from .qaoa_ref import QAOA

N_ANGLES = 2 * DEPTH


def tqa_units(s, p=DEPTH):
    """Trotterised-annealing schedule in normalised units.

    `s` is (N,) or a scalar; returns (N, 2p). gamma ramps 1/p -> 1 of `s`, beta ramps
    (1 - 1/p) -> 0, so the circuit interpolates from mixer-dominated to cost-dominated
    exactly as an annealing sweep does.
    """
    s = torch.as_tensor(s, dtype=torch.float32).reshape(-1, 1)
    l = torch.arange(1, p + 1, dtype=torch.float32, device=s.device).unsqueeze(0)
    return torch.cat([(l / p) * s, (1.0 - l / p) * s], dim=1)


def ramp_units(g0, g1, b0, b1, p=DEPTH):
    """A general linear schedule: gamma sweeps g0 -> g1, beta sweeps b0 -> b1.

    TQA is the special case (0, s, s, 0). Freeing the four endpoints keeps the family
    smooth — still 4 numbers rather than 10 — while letting a search adjust the ramp
    that TQA fixes by assumption.
    """
    t = torch.linspace(0, 1, p, device=g0.device).unsqueeze(0)
    g = g0.reshape(-1, 1) + (g1 - g0).reshape(-1, 1) * t
    b = b0.reshape(-1, 1) + (b1 - b0).reshape(-1, 1) * t
    return torch.cat([g, b], dim=1)


def sample_starts(n, kind, device, generator=None, s_range=(0.2, 4.0), jitter=0.25):
    """Starting points in normalised units.

    kind="uniform"  the old behaviour: 10 independent coordinates in the box.
    kind="tqa"      a TQA schedule with random `s`, plus Gaussian jitter.
    kind="ramp"     a random linear schedule (4 free endpoints), plus jitter.

    Jitter is what keeps these *starts* rather than a fixed guess — the optimiser still
    has to do the work, it just begins near the surface where the optima live.
    """
    def rnd(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    if kind == "uniform":
        return canonicalise(rnd(n, N_ANGLES) * 2 - 1)

    lo, hi = s_range
    if kind == "tqa":
        s = lo + (hi - lo) * rnd(n)
        u = tqa_units(s.cpu()).to(device)
    elif kind == "ramp":
        s = lo + (hi - lo) * rnd(n)
        u = ramp_units((rnd(n) * 0.4 - 0.2) * s, s * (0.6 + 0.8 * rnd(n)),
                       s * (0.6 + 0.8 * rnd(n)), (rnd(n) * 0.4 - 0.2) * s)
    else:
        raise ValueError(f"unknown init kind {kind!r}")

    noise = torch.randn(n, N_ANGLES, device=device, generator=generator)
    return canonicalise(u + jitter * noise)


@torch.no_grad()
def score(sim, h, u, scale, chunk=4096):
    out = []
    for lo in range(0, h.shape[0], chunk):
        a = to_angles(u[lo:lo + chunk], scale)
        out.append(sim.p_ground(h[lo:lo + chunk], a[:, :DEPTH], a[:, DEPTH:]))
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--points", type=int, default=200, help="values of s to scan")
    ap.add_argument("--s-min", type=float, default=0.02)
    ap.add_argument("--s-max", type=float, default=6.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                     device=dev)
    scale = angle_scale(sim, h, device=dev)
    n = h.shape[0]

    print(f"TQA schedule scan on {n} instances — no optimisation, forward passes only.")
    print(f"random angles score 1/4096 = {1 / 2 ** 12:.5f}\n")
    print(f"{'s':>7} {'mean P':>9} {'median':>9} {'max':>9}")

    best = (0.0, None)
    for s in torch.linspace(args.s_min, args.s_max, args.points):
        u = tqa_units(s.expand(n)).to(dev)
        p = score(sim, h, u, scale)
        m = p.mean().item()
        if m > best[0]:
            best = (m, s.item())
        if args.points <= 40 or int(s * 1000) % 250 < (args.s_max * 1000 / args.points):
            print(f"{s.item():7.3f} {m:9.5f} {p.median().item():9.5f} {p.max().item():9.5f}")

    m, s = best
    print(f"\nbest single s = {s:.4f}  ->  mean P(ground) = {m:.5f}")
    print("reference: proposer run 0.18405 | multistart nb 03 0.32396 | leaderboard #1 0.81468")
    print("\nNOTE: one scalar, zero optimiser steps, identical angles for every instance.")
    print("A constant submission scores 0 by the rules — this is a *starting point* test,")
    print("not a submission. What it measures is how much of the landscape the uniform")
    print("box init was throwing away.")
    return {"best_s": s, "mean_p": m}


if __name__ == "__main__":
    main()
