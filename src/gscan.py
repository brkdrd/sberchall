"""Is the gamma search box the reason we are stuck at 0.32?

Every search in this repo samples starting angles in the *normalised* units of
`angles.py`, where `gamma = u * 2*pi/span(E)` and `span(E) ~ 39`, so one unit of gamma
is 0.16 rad. `schedules.sample_starts` draws `s` from (0.2, 4.0), which caps every
restart at `|gamma| <= 0.64` rad. The legacy box was `(-pi, pi)`. So no search here has
ever evaluated a point with `|gamma| > pi`, and most evaluated none above 0.64.

The justification for that box — in `angles.py`'s own docstring — is that the phase
`exp(i*gamma*E)` completes a revolution by `gamma ~ 2*pi/span`, and "beyond that the
phase wraps and the landscape is chaotic rather than merely non-convex". That argument
prices the *span* of the spectrum. But the metric does not care about the span: it asks
for the probability of one specific state, so what has to be resolved is the **gap**
between the ground state and its nearest competitor. Measured on h_train:

    span(E)     ~ 39
    E1 - E0     median 0.17, 10th pct 0.033

A relative phase of order pi across that gap needs `gamma ~ pi/0.17 ~ 18` — two orders
of magnitude outside the box we have been searching, and 6x outside the legacy one. The
"chaos" the docstring warns about is, for this metric, where the signal has to live: a
phase that cannot tell E0 from E1 cannot concentrate amplitude on E0 alone.

That is a hypothesis, not a result, and this module is built to kill it or confirm it.
It sweeps the gamma sampling half-width over a geometric ladder, and at each rung runs
the same screen -> refine pipeline in **raw radians** with no box, no normalisation and
no schedule prior. Two numbers per rung decide it: the best P(ground) reached, and the
mean |gamma| the refined winners actually settled on. If the winners at every rung drift
back to |gamma| < 1 and P stalls near 0.33, the box was never the problem and the p=5
ceiling is real. If P climbs with the rung, we have been searching the wrong region.

Everything is evaluated by the organisers' `QAOA.p_ground`, unmodified and uncached, so
no reimplementation sits between the diagnostic and the score.

Usage:
    python -m src.gscan --instances 8 --starts 4096          # the diagnostic
    python -m src.gscan --instances 8 --scales 8,16,32 --deep 3000
    python -m src.gscan --instances 16 --full                 # diagnose, then submit
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .predict import write_submission
from .qaoa_ref import QAOA, P as DEPTH

N_ANGLES = 2 * DEPTH
LADDER = [0.16, 0.64, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
INITS = ("uniform", "geom")


def sample_start_angles(kind, n, gmax, gen, device):
    """Starting angles in radians, with |gamma| <= gmax.

    kind="uniform"  five independent gammas in the box — what a box sweep means.
    kind="geom"     a *geometric* gamma ladder across the layers. If the hypothesis
                    above is right, one gamma cannot do the whole job: funnelling
                    amplitude into the low-energy region wants `gamma ~ 1/(local field)
                    ~ 1`, while separating E0 from E1 wants `gamma ~ pi/(E1-E0) ~ 18`.
                    Those are different jobs at different scales, and five layers can
                    do them in sequence — coarse first, fine last — which is a
                    two-parameter family (top scale and ratio) inside the box, not a
                    five-dimensional volume to be sampled. `beta` ramps down, as in
                    every smooth QAOA schedule.
    """
    def rnd(*shape):
        return torch.rand(*shape, device=device, generator=gen)

    if kind == "uniform":
        g = (rnd(n, DEPTH) * 2 - 1) * gmax
        b = (rnd(n, DEPTH) - 0.5) * torch.pi
        return torch.cat([g, b], dim=1)

    if kind == "geom":
        top = gmax * (0.3 + 0.7 * rnd(n, 1))
        ratio = 1.2 + 2.8 * rnd(n, 1)
        l = torch.arange(DEPTH - 1, -1, -1, dtype=torch.float32, device=device)
        g = top / ratio ** l
        g = g * torch.where(rnd(n, 1) < 0.5, -1.0, 1.0)
        b0 = (rnd(n, 1) - 0.5) * torch.pi
        ramp = 1.0 - torch.arange(DEPTH, dtype=torch.float32, device=device) / DEPTH
        b = b0 * ramp + 0.1 * torch.randn(n, DEPTH, device=device, generator=gen)
        return torch.cat([g, b], dim=1)

    raise ValueError(f"unknown init {kind!r}")


@torch.no_grad()
def score(sim, h, ang, chunk=4096):
    """P(ground) for a batch of (h, angles) rows, via the organisers' simulator."""
    out = []
    for lo in range(0, ang.shape[0], chunk):
        a, hh = ang[lo:lo + chunk], h[lo:lo + chunk]
        out.append(sim.p_ground(hh, a[:, :DEPTH], a[:, DEPTH:]))
    return torch.cat(out)


def refine(sim, h, ang, steps, lr_gamma, lr_beta, chunk=1024):
    """Adam on the angles in radians. Returns the best point *visited*, not the last.

    P(ground) is not monotone along an Adam path, so keeping the final iterate throws
    away better points the path passed through. Separate step sizes for the two halves
    because a step meaningful for beta is invisible to a gamma of order 10.
    """
    best_ang, best_p = ang.clone(), score(sim, h, ang, chunk)
    for lo in range(0, ang.shape[0], chunk):
        sl = slice(lo, min(lo + chunk, ang.shape[0]))
        g = ang[sl, :DEPTH].clone().requires_grad_(True)
        b = ang[sl, DEPTH:].clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [g], "lr": lr_gamma},
                                {"params": [b], "lr": lr_beta}])
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            p = sim.p_ground(h[sl], g, b)
            (-p.clamp_min(1e-30).log()).sum().backward()
            opt.step()
            with torch.no_grad():
                cur = torch.cat([g, b], dim=1)
                better = p.detach() > best_p[sl]
                if better.any():
                    idx = torch.nonzero(better, as_tuple=True)[0]
                    best_p[sl.start + idx] = p.detach()[idx]
                    best_ang[sl.start + idx] = cur[idx]
    return best_ang, best_p


def rung(sim, h, inst, n_starts, top, steps, lr, gmax, gen, device, chunk,
         init="uniform"):
    """One ladder rung: screen `n_starts` points with |gamma| <= gmax, refine the best."""
    n = inst.shape[0]
    hrep = h[inst].repeat_interleave(n_starts, dim=0)
    ang = sample_start_angles(init, n * n_starts, gmax, gen, device)

    p = score(sim, hrep, ang, chunk * 8)
    keep = p.view(n, n_starts).topk(top, dim=1).indices
    keep = keep + torch.arange(n, device=device).unsqueeze(1) * n_starts
    keep = keep.reshape(-1)

    hk, ak = hrep[keep], ang[keep]
    ak, pk = refine(sim, hk, ak, steps, lr * gmax, lr * torch.pi / 2, chunk)

    pk = pk.view(n, top)
    win = pk.argmax(dim=1)
    best_ang = ak.view(n, top, N_ANGLES)[torch.arange(n, device=device), win]
    return {
        "gmax": gmax,
        "init": init,
        "screen_best": p.view(n, n_starts).max(dim=1).values,
        "best": pk.max(dim=1).values,
        "gamma_absmean": best_ang[:, :DEPTH].abs().mean().item(),
        "gamma_absmax": best_ang[:, :DEPTH].abs().max().item(),
        "angles": best_ang,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=None, help="npz with per-rung results")
    ap.add_argument("--instances", type=int, default=8, help="first N rows of h_train")
    ap.add_argument("--starts", type=int, default=4096, help="screened points per rung")
    ap.add_argument("--top", type=int, default=32, help="screened points refined per rung")
    ap.add_argument("--steps", type=int, default=400, help="Adam steps per refined point")
    ap.add_argument("--lr", type=float, default=0.02,
                    help="step size as a fraction of each half's range")
    ap.add_argument("--inits", default=",".join(INITS),
                    help="comma-separated start families (see sample_start_angles)")
    ap.add_argument("--scales", default=None,
                    help="comma-separated gamma half-widths (default: the geometric ladder)")
    ap.add_argument("--full", action="store_true",
                    help="after the ladder, run the winning rung over all of h_train "
                         "and write a submission (the leaderboard is scored on h_train)")
    ap.add_argument("--full-starts", type=int, default=16384)
    ap.add_argument("--full-top", type=int, default=64)
    ap.add_argument("--full-steps", type=int, default=400)
    ap.add_argument("--full-block", type=int, default=50,
                    help="instances searched per block, to bound VRAM")
    ap.add_argument("--deep", type=int, default=0,
                    help="extra Adam steps on the winning point of the winning rung")
    ap.add_argument("--chunk", type=int, default=1024, help="rows refined at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                     device=dev)
    inst = torch.arange(min(args.instances, h.shape[0]), device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    scales = ([float(s) for s in args.scales.split(",")] if args.scales else LADDER)

    E = sim.energies(h[inst])
    Es = E.sort(dim=1).values
    print(f"gamma-box sweep on {len(inst)} instances of h_train, {args.starts} screened "
          f"+ {args.top} refined x {args.steps} Adam steps per rung.")
    print(f"device {dev} | span(E) {(Es[:, -1] - Es[:, 0]).mean():.2f} | "
          f"E1-E0 median {(Es[:, 1] - Es[:, 0]).median():.4f} | "
          f"pi/(E1-E0) median {(torch.pi / (Es[:, 1] - Es[:, 0])).median():.1f}")
    print(f"random angles score 1/4096 = {1 / 4096:.5f}; our best submission = 0.32\n")
    print(f"{'init':>8} {'|gamma|<=':>10} {'screen max':>11} {'refined mean':>13} "
          f"{'refined max':>12} {'min':>8} {'|g| of winners':>15}")

    results, best_rung = {}, None
    inits = args.inits.split(",")
    for init in inits:
        for gmax in scales:
            t0 = time.time()
            r = rung(sim, h, inst, args.starts, args.top, args.steps, args.lr, gmax,
                     gen, dev, args.chunk, init=init)
            print(f"{init:>8} {gmax:10.2f} {r['screen_best'].max().item():11.5f} "
                  f"{r['best'].mean().item():13.5f} {r['best'].max().item():12.5f} "
                  f"{r['best'].min().item():8.5f} {r['gamma_absmean']:15.2f}"
                  f"   [{time.time() - t0:.0f}s]")
            results[f"best_{init}_{gmax}"] = r["best"].cpu().numpy()
            results[f"angles_{init}_{gmax}"] = r["angles"].cpu().numpy()
            if best_rung is None or r["best"].mean() > best_rung["best"].mean():
                best_rung = r

    print(f"\nbest rung: init={best_rung['init']}, |gamma| <= {best_rung['gmax']}, "
          f"mean P = {best_rung['best'].mean().item():.5f}, max P = "
          f"{best_rung['best'].max().item():.5f}")

    if args.deep:
        ang, p = refine(sim, h[inst], best_rung["angles"], args.deep,
                        args.lr * best_rung["gmax"] * 0.2, args.lr * torch.pi / 2 * 0.2,
                        args.chunk)
        print(f"after {args.deep} more Adam steps: mean P = {p.mean().item():.5f}, "
              f"max = {p.max().item():.5f}, min = {p.min().item():.5f}")
        print("per instance: " + " ".join(f"{x:.3f}" for x in p.tolist()))
        results["deep_p"] = p.cpu().numpy()
        results["deep_angles"] = ang.cpu().numpy()

    if args.full:
        gmax, init = best_rung["gmax"], best_rung["init"]
        print(f"\nfull pass over {h.shape[0]} instances at init={init}, "
              f"|gamma| <= {gmax}: "
              f"{args.full_starts} screened + {args.full_top} refined x "
              f"{args.full_steps} steps each")
        t0, ang_all, p_all = time.time(), [], []
        for lo in range(0, h.shape[0], args.full_block):
            block = torch.arange(lo, min(lo + args.full_block, h.shape[0]), device=dev)
            r = rung(sim, h, block, args.full_starts, args.full_top, args.full_steps,
                     args.lr, gmax, gen, dev, args.chunk, init=init)
            ang_all.append(r["angles"])
            p_all.append(r["best"])
            done = lo + len(block)
            print(f"  {done:4d}/{h.shape[0]}  running mean P = "
                  f"{torch.cat(p_all).mean().item():.5f}  [{time.time() - t0:.0f}s]")
        ang_all, p_all = torch.cat(ang_all), torch.cat(p_all)
        print(f"full pass: mean P(ground) = {p_all.mean().item():.5f}, "
              f"median {p_all.median().item():.5f}, min {p_all.min().item():.5f}, "
              f"max {p_all.max().item():.5f}  [{(time.time() - t0) / 60:.1f} min]")
        results["full_p"] = p_all.cpu().numpy()
        results["full_angles"] = ang_all.cpu().numpy()
        if args.out:
            write_submission(args.out.with_name("submission_train.csv"), ang_all)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, scales=np.array(scales), instances=inst.cpu().numpy(),
                 **results)
        print(f"wrote {args.out}")
    out = {"best_gmax": best_rung["gmax"], "best_init": best_rung["init"],
           "best_mean_p": best_rung["best"].mean().item()}
    if args.full:
        out["full_mean_p"] = float(results["full_p"].mean())
    return out


if __name__ == "__main__":
    main()
