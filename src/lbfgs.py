"""Batched L-BFGS over the angles: 500+ independent quasi-Newton searches in lockstep.

Adam is in this repo by inheritance, not by argument. It exists to survive *noisy*
minibatch gradients — it estimates gradient moments to average that noise away. Ours are
exact and deterministic: the simulator is differentiable and there is no sampling anywhere.
So Adam's conservatism buys nothing here, and we pay for it with a fixed step size, no
curvature information, and hundreds of iterations per restart.

`torch.optim.LBFGS` is not a drop-in replacement. It optimises *one* problem: it would see
our 500 instances as a single parameter vector and drive its line search off the summed
objective, handing every instance a step size chosen for the aggregate. So L-BFGS is
implemented here directly, vectorised over the batch — the two-loop recursion is a handful
of batched dot products, and the memory pairs are a (B, m, 10) tensor.

Three things this buys beyond faster convergence:

- **Asynchronous restarts.** Instances converge at different times. Rather than running a
  fixed restarts x steps grid — where converged instances sit idle burning evaluations —
  each instance restarts the moment it converges, from a fresh start with cleared memory.
  Every evaluation stays productive.
- **A real stopping rule.** Adam runs for a step count you guessed. L-BFGS stops on the
  gradient norm or a failed line search, so "converged" is measured, not assumed.
- **More restarts per unit budget.** This is the real prize. The landscape is multi-modal,
  so what finds a good basin is *trying more basins*, not more precision inside one. If a
  restart finishes in 30 iterations instead of 300 Adam steps, the same budget buys an
  order of magnitude more attempts.

Curvature pairs use the cautious update of Li & Fukushima: a pair is stored only when
s.y is safely positive, which keeps the implicit Hessian positive definite without needing
a strong-Wolfe line search. Backtracking Armijo is enough, and is trivially batchable —
each instance keeps its own step length and they halve independently under a mask.

Budget is counted in **forward passes** so it is comparable with the other methods here
(notebook 03 spends ~152k per instance): a value costs 1, a value-and-gradient costs 2.

Usage:
    python -m src.lbfgs --budget 20000 --out runs/lbfgs_submission.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .angles import DEPTH, angle_scale, to_angles
from .predict import write_submission
from .qaoa_ref import QAOA
from .rollout import log_p
from .schedules import sample_starts

N_ANGLES = 2 * DEPTH


class Budget:
    """Forward-pass accounting, so every method in this repo is priced in one currency."""

    def __init__(self, n):
        self.n, self.fwd = n, 0

    def per_instance(self):
        return self.fwd // self.n


def f_only(sim, h, u, scale, bud, chunk):
    """Objective only: minimise -log P(ground). Costs one forward pass."""
    out = []
    with torch.no_grad():
        for i in range(0, u.shape[0], chunk):
            out.append(-log_p(sim, h[i:i + chunk], u[i:i + chunk], scale))
    bud.fwd += u.shape[0]
    return torch.cat(out)


def f_and_g(sim, h, u, scale, bud, chunk):
    """Objective and its exact gradient. Priced at two forward passes."""
    fs, gs = [], []
    for i in range(0, u.shape[0], chunk):
        uu = u[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            lp = log_p(sim, h[i:i + chunk], uu, scale)
            (g,) = torch.autograd.grad(lp.sum(), uu)
        fs.append(-lp.detach())
        gs.append(-g)
    bud.fwd += 2 * u.shape[0]
    return torch.cat(fs), torch.cat(gs)


class Memory:
    """L-BFGS curvature pairs, newest in the last slot so the two-loop order is uniform.

    A ring buffer would be cheaper, but its ordering depends on a per-instance pointer,
    and the two-loop recursion has to visit pairs newest-to-oldest for every instance at
    once. Rolling keeps slot m-1 newest for everyone; at (B, 10, 10) the copy is free.
    """

    def __init__(self, b, m, d, device, dtype):
        self.S = torch.zeros(b, m, d, device=device, dtype=dtype)
        self.Y = torch.zeros(b, m, d, device=device, dtype=dtype)
        self.rho = torch.zeros(b, m, device=device, dtype=dtype)
        self.ok = torch.zeros(b, m, dtype=torch.bool, device=device)
        self.m = m

    def push(self, s, y, sy, mask):
        """Store (s, y) for the masked instances; roll everyone so slot -1 is newest."""
        for t in (self.S, self.Y):
            t[:, :-1] = t[:, 1:].clone()
        self.rho[:, :-1] = self.rho[:, 1:].clone()
        self.ok[:, :-1] = self.ok[:, 1:].clone()
        self.S[:, -1] = torch.where(mask.unsqueeze(-1), s, torch.zeros_like(s))
        self.Y[:, -1] = torch.where(mask.unsqueeze(-1), y, torch.zeros_like(y))
        self.rho[:, -1] = torch.where(mask, 1.0 / sy.clamp_min(1e-12),
                                      torch.zeros_like(sy))
        self.ok[:, -1] = mask

    def clear(self, mask):
        self.ok[mask] = False
        self.rho[mask] = 0.0

    def direction(self, g):
        """Two-loop recursion. Invalid slots carry rho = 0 and are exact no-ops."""
        q = g.clone()
        alpha = torch.zeros(g.shape[0], self.m, device=g.device, dtype=g.dtype)
        for i in range(self.m - 1, -1, -1):
            a = self.rho[:, i] * (self.S[:, i] * q).sum(-1)
            alpha[:, i] = a
            q = q - a.unsqueeze(-1) * self.Y[:, i]
        # Nocedal's H0 scaling from the newest pair; falls back to steepest descent
        sy = (self.S[:, -1] * self.Y[:, -1]).sum(-1)
        yy = (self.Y[:, -1] * self.Y[:, -1]).sum(-1)
        gamma = torch.where(self.ok[:, -1], sy / yy.clamp_min(1e-12),
                            torch.ones_like(sy))
        r = gamma.clamp(1e-8, 1e8).unsqueeze(-1) * q
        for i in range(self.m):
            b = self.rho[:, i] * (self.Y[:, i] * r).sum(-1)
            r = r + self.S[:, i] * (alpha[:, i] - b).unsqueeze(-1)
        return -r


def armijo(sim, h, scale, bud, chunk, x, f, g, d, t0, c1=1e-4, max_ls=15):
    """Backtracking line search, each instance halving its own step under a mask.

    Returns (step, f_new, accepted). Instances that never satisfy Armijo are reported as
    failures — that is the signal to declare them converged and restart them.
    """
    gd = (g * d).sum(-1)
    t = t0.clone()
    best_t = torch.zeros_like(t)
    best_f = f.clone()
    done = torch.zeros_like(t, dtype=torch.bool)
    for _ in range(max_ls):
        # evaluate only the instances still backtracking: an instance that has already
        # accepted a step must not be re-measured, and the budget must not be charged for it
        idx = (~done).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        ft = f_only(sim, h[idx], x[idx] + t[idx].unsqueeze(-1) * d[idx], scale, bud, chunk)
        hit = (ft <= f[idx] + c1 * t[idx] * gd[idx]) & torch.isfinite(ft)
        sel = idx[hit]
        best_t[sel] = t[sel]
        best_f[sel] = ft[hit]
        done[sel] = True
        t[idx[~hit]] = t[idx[~hit]] * 0.5
    return best_t, best_f, done


def search(sim, h, scale, args, gen):
    """Run L-BFGS with asynchronous restarts until the forward-pass budget is spent."""
    dev, n = h.device, h.shape[0]
    bud = Budget(n)
    chunk = args.chunk
    mem = Memory(n, args.memory, N_ANGLES, dev, torch.float32)

    x = sample_starts(n, args.init, dev, gen)
    f, g = f_and_g(sim, h, x, scale, bud, chunk)
    best_f, best_x = f.clone(), x.clone()
    iters = torch.zeros(n, device=dev)
    restarts = torch.zeros(n, device=dev)
    conv_iters, ls_evals, ls_calls = [], 0, 0
    why = {"gtol": 0, "ls": 0, "cap": 0}
    curve, t0 = [], time.time()
    target = args.budget * n
    deadline = args.max_hours * 3600 if args.max_hours else None

    while bud.fwd < target:
        if deadline and time.time() - t0 > deadline:
            print(f"  wall-clock cap reached at {bud.per_instance()} fwd/instance "
                  f"-- keeping the best point found", flush=True)
            break
        d = mem.direction(g)
        # a fresh restart has no curvature information: cap the first step by the gradient
        fresh = ~mem.ok[:, -1]
        t0v = torch.where(fresh, (1.0 / g.abs().sum(-1).clamp_min(1e-8)).clamp(max=1.0),
                          torch.ones_like(f))
        before = bud.fwd
        t, f_new_ls, ok = armijo(sim, h, scale, bud, chunk, x, f, g, d, t0v,
                                 max_ls=args.max_ls)
        ls_evals += bud.fwd - before
        ls_calls += 1

        step = t.unsqueeze(-1) * d
        x_new = x + step
        f_new, g_new = f_and_g(sim, h, x_new, scale, bud, chunk)

        # cautious update (Li & Fukushima): only store pairs with safely positive curvature
        s, y = step, g_new - g
        sy = (s * y).sum(-1)
        good = ok & (sy > 1e-10 * s.norm(dim=-1) * y.norm(dim=-1))
        mem.push(s, y, sy, good)

        adv = ok.unsqueeze(-1)
        x = torch.where(adv, x_new, x)
        f = torch.where(ok, f_new, f)
        g = torch.where(adv, g_new, g)
        iters += ok.float()

        better = f_new < best_f
        best_f = torch.where(better, f_new, best_f)
        best_x = torch.where(better.unsqueeze(-1), x_new, best_x)

        # converged: flat gradient, a line search that found no acceptable step, or a cap
        flat = g.abs().amax(-1) < args.gtol
        capped = iters >= args.max_iters
        done = flat | (~ok) | capped
        if bool(done.any()):
            why["gtol"] += int((flat & done).sum())
            why["ls"] += int(((~ok) & done & ~flat).sum())
            why["cap"] += int((capped & done & ~flat & ok).sum())
            conv_iters.append(iters[done].mean().item())
            k = int(done.sum())
            fresh_x = sample_starts(k, args.init, dev, gen)
            x = x.clone(); x[done] = fresh_x
            mem.clear(done)
            iters[done] = 0
            restarts[done] += 1
            fk, gk = f_and_g(sim, h[done], fresh_x, scale, bud, chunk)
            f = f.clone(); f[done] = fk
            g = g.clone(); g[done] = gk

        if len(curve) == 0 or bud.per_instance() - curve[-1][0] >= args.log_every:
            p = (-best_f).exp()
            curve.append((bud.per_instance(), p.mean().item(), p.median().item(),
                          time.time() - t0))
            print(f"  fwd/inst {bud.per_instance():7d}  mean P {p.mean():.4f}  "
                  f"median {p.median():.4f}  restarts/inst {restarts.mean():5.1f}  "
                  f"{time.time() - t0:6.0f}s", flush=True)

    stats = {"restarts_mean": restarts.mean().item(),
             "conv_gtol": int(why["gtol"]), "conv_ls_fail": int(why["ls"]),
             "conv_cap": int(why["cap"]),
             "iters_to_converge": float(np.mean(conv_iters)) if conv_iters else float("nan"),
             "ls_evals_per_iter": ls_evals / max(ls_calls * n, 1)}
    return best_x, (-best_f).exp(), bud, curve, stats


def adam_matched(sim, h, scale, args, gen, target_fwd):
    """Adam multistart held to the same forward-pass budget, as the control."""
    dev, n = h.device, h.shape[0]
    bud = Budget(n)
    best = torch.zeros(n, device=dev)
    while bud.fwd < target_fwd:
        u = sample_starts(n, args.init, dev, gen).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=args.adam_lr)
        for _ in range(args.adam_steps):
            opt.zero_grad()
            lp = log_p(sim, h, u, scale)
            (-lp.sum()).backward()
            bud.fwd += 2 * n
            with torch.no_grad():
                best = torch.maximum(best, lp.detach().exp())
            opt.step()
            if bud.fwd >= target_fwd:
                break
    return best, bud


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--h", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/lbfgs_submission.csv"))
    ap.add_argument("--budget", type=int, default=20000,
                    help="forward passes per instance (value=1, value+grad=2)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock cap; 0 = unlimited. The best point so far is kept.")
    ap.add_argument("--memory", type=int, default=10, help="L-BFGS curvature pairs")
    ap.add_argument("--max-iters", type=int, default=200, help="cap per restart")
    ap.add_argument("--max-ls", type=int, default=15, help="Armijo backtracks")
    ap.add_argument("--gtol", type=float, default=1e-5)
    ap.add_argument("--init", default="ramp", choices=("uniform", "tqa", "ramp"))
    ap.add_argument("--adam-lr", type=float, default=0.03)
    ap.add_argument("--adam-steps", type=int, default=300)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--log-every", type=int, default=1000, help="fwd/instance between logs")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h = torch.tensor(np.load(args.h or (args.data_dir / "h_train.npy")),
                     dtype=torch.float32, device=dev)
    scale = angle_scale(sim, torch.tensor(np.load(args.data_dir / "h_train.npy"),
                                          dtype=torch.float32, device=dev), device=dev)

    print(f"batched L-BFGS | {h.shape[0]} instances | {args.budget} forward passes each | "
          f"memory {args.memory} | init {args.init} | {dev}", flush=True)
    t0 = time.time()
    u, p, bud, curve, st = search(sim, h, scale, args, gen)
    secs = time.time() - t0

    print(f"\nL-BFGS          : mean P {p.mean():.5f}  median {p.median():.5f}  "
          f"({bud.per_instance()} fwd/inst, {secs:.0f}s)")
    print(f"  restarts/instance      {st['restarts_mean']:.1f}")
    print(f"  iterations to converge {st['iters_to_converge']:.1f}")
    print(f"  line-search evals/iter {st['ls_evals_per_iter']:.2f}  "
          f"(1.0 = step accepted first try)")
    print(f"  restarts ended by: flat gradient {st['conv_gtol']}, "
          f"line-search failure {st['conv_ls_fail']}, iteration cap {st['conv_cap']}")

    if not args.skip_control:
        pa, ba = adam_matched(sim, h, scale, args, gen, bud.fwd)
        print(f"Adam, same budget: mean P {pa.mean():.5f}  median {pa.median():.5f}  "
              f"({ba.per_instance()} fwd/inst)")
        print(f"  -> L-BFGS is {p.mean().item() / max(pa.mean().item(), 1e-12):.3f}x Adam "
              f"at matched cost")

    lv = [10, 25, 50, 75, 90]
    qq = torch.quantile(p, torch.tensor([l / 100 for l in lv], device=dev))
    print("\nper-instance P(ground): " +
          "  ".join(f"p{l:02d} {v:.4f}" for l, v in zip(lv, qq.tolist())) +
          f"  max {p.max().item():.4f}")
    if curve:
        print("\nscaling curve")
        print(f"   {'fwd/inst':>9} {'mean P':>9} {'median':>9} {'sec':>7}")
        for e, m, md, t in curve[::max(1, len(curve) // 12)] + [curve[-1]]:
            print(f"   {e:>9} {m:>9.4f} {md:>9.4f} {t:>7.0f}")

    print(f"\ntotal {secs:.0f}s — {'within' if secs < 600 else 'OVER'} the 10-minute limit")
    print("reference: proposer 0.18405 | multistart nb 03 0.32396 | leaderboard #1 0.81468")
    write_submission(args.out, to_angles(u, scale))
    return {"submission": args.out, "mean_p": p.mean().item(),
            "median_p": p.median().item(), "max_p": p.max().item(),
            "seconds": secs, "fwd_per_instance": bud.per_instance(),
            "quantiles": dict(zip([f"p{l}" for l in lv], qq.tolist())),
            "curve": curve, **st}


if __name__ == "__main__":
    main()
