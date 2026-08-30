"""Anytime profiling: how much budget does each instance actually need?

The question this answers is not "what score does the method reach" but "how is that score
distributed over instances and over budget" — which instances are solved in the first fifty
Adam steps, which need thousands, and which never get there no matter what. That is what
decides where to stop spending.

**Budget is counted in Adam steps per instance, not seconds.** Everything here runs batched:
500 instances advance in lockstep on one GPU, so no instance has a wall-clock time of its
own. Steps-per-instance is hardware-independent and comparable across runs; the measured
throughput is reported alongside so it converts back to seconds for the whole 500-instance
pass, which is the form the 10-minute inference limit is written in.

Each restart runs Adam from a fresh start and the running best is recorded at *every* step,
which is free: `log_p` is already computed for the gradient. So one run yields a full
anytime curve per instance rather than a single endpoint.

Instances are `h_train` plus freshly generated ones. `h_train` is i.i.d. U(-1,1)
(KS-verified), so generated instances are drawn from the same law — and the two groups are
reported separately, which doubles as a check that they really do behave alike.

Usage:
    python -m src.anytime --extra 1500 --restarts 20 --steps 300
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .angles import DEPTH, angle_scale, to_angles
from .qaoa_ref import QAOA
from .rollout import log_p
from .schedules import sample_starts

SHADE = " .:-=+*#%@"


def anytime_block(sim, h, scale, kind, restarts, steps, lr, gen, deadline=None, t0=None):
    """Running best P(ground) per instance at every Adam step. Returns (N, restarts*steps)."""
    n = h.shape[0]
    best = torch.zeros(n, device=h.device)
    out = torch.zeros(n, restarts * steps, device=h.device)
    k = 0
    for r in range(restarts):
        u = sample_starts(n, kind, h.device, gen).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            lp = log_p(sim, h, u, scale)
            (-lp.sum()).backward()
            with torch.no_grad():
                best = torch.maximum(best, lp.detach().exp())
                out[:, k] = best
            k += 1
            opt.step()
        if deadline and time.time() - t0 > deadline:
            out = out[:, :k]
            break
    return out


def time_to(curves, thresh):
    """First step index at which each instance reaches `thresh`; -1 if it never does."""
    hit = curves >= thresh
    any_hit = hit.any(dim=1)
    first = hit.float().argmax(dim=1)
    return torch.where(any_hit, first, torch.full_like(first, -1)), any_hit


def heatmap(curves, budgets, n_q=18, n_b=26, qmax=None):
    """Text heatmap: budget on x, achieved quality on y, cell = share of instances."""
    n, t = curves.shape
    cols = np.unique(np.geomspace(1, t, n_b).astype(int)) - 1
    qmax = qmax or float(np.quantile(curves[:, -1], 0.995))
    edges = np.linspace(0, max(qmax, 1e-6), n_q + 1)
    grid = np.zeros((n_q, len(cols)))
    for j, c in enumerate(cols):
        idx = np.clip(np.digitize(curves[:, c], edges) - 1, 0, n_q - 1)
        grid[:, j] = np.bincount(idx, minlength=n_q) / n

    lines = []
    top = grid.max() or 1.0
    for i in range(n_q - 1, -1, -1):
        row = "".join(SHADE[min(len(SHADE) - 1, int(v / top * (len(SHADE) - 1) + 0.5))]
                      for v in grid[i])
        lines.append(f"  {edges[i]:5.3f}-{edges[i + 1]:5.3f} |{row}|")
    lines.append("  " + " " * 13 + "+" + "-" * len(cols) + "+")
    lines.append("  " + " " * 14 + f"{budgets[cols[0]]} steps"
                 + " " * max(1, len(cols) - 18) + f"{budgets[cols[-1]]} steps")
    return "\n".join(lines), grid, cols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("runs/anytime.npz"))
    ap.add_argument("--extra", type=int, default=1500, help="generated instances to add")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300, help="Adam steps per restart")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--init", default="tqa", choices=("uniform", "tqa", "ramp"))
    ap.add_argument("--block", type=int, default=1000, help="instances held on GPU at once")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h_tr = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                        device=dev)
    scale = angle_scale(sim, h_tr, device=dev)
    extra = torch.rand(args.extra, h_tr.shape[1], device=dev, generator=gen) * 2 - 1
    h = torch.cat([h_tr, extra])
    is_train = torch.zeros(len(h), dtype=torch.bool)
    is_train[:len(h_tr)] = True

    total = args.restarts * args.steps
    print(f"anytime profile: {len(h)} instances ({len(h_tr)} train + {args.extra} generated)"
          f"\n  init={args.init}  {args.restarts} restarts x {args.steps} Adam steps "
          f"= {total} steps/instance  |  {dev}", flush=True)

    t0 = time.time()
    deadline = args.max_hours * 3600 if args.max_hours else None
    parts = []
    for lo in range(0, len(h), args.block):
        hb = h[lo:lo + args.block]
        c = anytime_block(sim, hb, scale, args.init, args.restarts, args.steps, args.lr,
                          gen, deadline, t0)
        parts.append(c.cpu())
        el = time.time() - t0
        print(f"  block {lo // args.block + 1}/{-(-len(h) // args.block)}  "
              f"{len(hb)} instances  mean best P {c[:, -1].mean():.4f}  {el:.0f}s", flush=True)
        if deadline and el > deadline:
            print("  wall-clock budget reached -- profiling what is complete", flush=True)
            break

    width = min(p.shape[1] for p in parts)
    curves = torch.cat([p[:, :width] for p in parts]).numpy()
    n_done = curves.shape[0]
    is_train = is_train[:n_done].numpy()
    budgets = np.arange(1, width + 1)
    secs = time.time() - t0
    # cost of one Adam step for a 500-instance batch, derived from total instance-steps
    # so it stays correct however the run was blocked
    per_step_500 = 500.0 * secs / max(n_done * width, 1)

    print(f"\nthroughput: {width} steps in {secs:.0f}s across {n_done} instances")
    print(f"  -> one 500-instance pass costs about {per_step_500:.3f} s per Adam step,")
    print(f"     so the 600 s inference limit buys roughly "
          f"{int(600 / max(per_step_500, 1e-9))} steps/instance\n")

    print("HEATMAP — share of instances at each quality, as budget grows")
    print("  (column = Adam steps/instance, log-spaced; row = best P(ground) so far)\n")
    art, grid, cols = heatmap(curves, budgets)
    print(art)

    print("\nMEAN / MEDIAN best P vs budget")
    print(f"  {'steps':>7} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8} "
          f"{'d(mean)/doubling':>18}")
    marks = [c for c in np.unique(np.geomspace(1, width, 14).astype(int)) - 1]
    prev = None
    for c in marks:
        col = curves[:, c]
        d = "" if prev is None else f"{col.mean() - prev:+.4f}"
        print(f"  {budgets[c]:>7} {col.mean():>8.4f} {np.median(col):>8.4f} "
              f"{np.quantile(col, .1):>8.4f} {np.quantile(col, .9):>8.4f} {d:>18}")
        prev = col.mean()

    print("\nTIME TO REACH A THRESHOLD (Adam steps/instance)")
    print(f"  {'target P':>9} {'reached':>9} {'p25':>8} {'median':>8} {'p75':>8}")
    ct = torch.from_numpy(curves)
    for th in (0.05, 0.10, 0.20, 0.30, 0.50):
        first, ok = time_to(ct, th)
        if ok.any():
            f = first[ok].float()
            print(f"  {th:>9.2f} {ok.float().mean().item():>8.1%} "
                  f"{f.quantile(.25).item():>8.0f} {f.median().item():>8.0f} "
                  f"{f.quantile(.75).item():>8.0f}")
        else:
            print(f"  {th:>9.2f} {'0.0%':>8}")

    print("\nWHEN TO STOP — share of instances already within 95% of their own final value")
    for c in marks:
        frac = (curves[:, c] >= 0.95 * curves[:, -1]).mean()
        print(f"  {budgets[c]:>7} steps: {frac:>6.1%}")

    tr, ge = curves[is_train][:, -1], curves[~is_train][:, -1]
    if len(ge):
        print(f"\nsanity — train vs generated final P: {tr.mean():.4f} vs {ge.mean():.4f} "
              f"(should match; both are U(-1,1))")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, curves=curves.astype(np.float32), budgets=budgets,
                        is_train=is_train, grid=grid, cols=cols,
                        sec_per_step_500=per_step_500, init=args.init, lr=args.lr,
                        steps=args.steps, restarts=args.restarts)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB) "
          f"— full per-instance curves, for plotting outside this run")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        ax[0].imshow(grid, origin="lower", aspect="auto", cmap="magma",
                     extent=[0, len(cols), 0, float(np.quantile(curves[:, -1], .995))])
        ax[0].set_xlabel("budget (log-spaced Adam steps/instance)")
        ax[0].set_ylabel("best P(ground) so far")
        ax[0].set_title("share of instances by quality vs budget")
        q = [np.quantile(curves, p, axis=0) for p in (.1, .25, .5, .75, .9)]
        for lab, v in zip(("p10", "p25", "median", "p75", "p90"), q):
            ax[1].plot(budgets, v, label=lab)
        ax[1].set_xscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[1].set_xlabel("Adam steps/instance"); ax[1].set_ylabel("best P(ground)")
        ax[1].set_title("anytime quantiles")
        fig.tight_layout()
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(no plot: {e})")

    return {"curves": str(args.out), "n_instances": n_done, "steps": int(width),
            "mean_final": float(curves[:, -1].mean()),
            "sec_per_step_500": float(per_step_500)}


if __name__ == "__main__":
    main()
