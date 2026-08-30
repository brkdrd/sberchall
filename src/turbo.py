"""TuRBO for QAOA angles: 500 trust-region Bayesian optimisers advancing in lockstep.

TuRBO (Eriksson, Pearce, Gardner, Turner & Poloczek, NeurIPS 2019) exists because a single
global GP is the wrong model for a rugged landscape: it over-smooths, and its acquisition
function ends up sampling the mean rather than any optimum. Instead it keeps a *local* GP
inside a trust region centred on the best point found, picks candidates by Thompson
sampling, and grows or shrinks the region on streaks of success or failure — restarting it
elsewhere when it collapses.

That is a good fit for this problem's pathology. The angle landscape is violently
multi-modal: over probe instances, best-of-24 restarts reached P ~ 0.33 while the median
restart reached 0.13. Adam is excellent once it is in the right basin and useless at
choosing one, which is exactly the division of labour a trust region is built for.

**The engineering that makes it viable here.** Textbook BO assumes evaluations are
expensive and serial. Ours are cheap, differentiable and batched — but there are 500
independent problems, one per instance of `h`. So this runs `N x n_tr` independent trust
regions simultaneously, with one batched Cholesky per step instead of a Python loop, the
same trick notebook 04 used for CMA-ES.

**What is state of the art here beyond stock TuRBO**

- **Initial design is TQA-informed, not just Sobol.** Optimal QAOA angles lie near a smooth
  annealing schedule (see `schedules.py`), so seeding part of the design from that family
  puts trust regions near the right surface from step one instead of spending their budget
  finding it.
- **The domain is symmetry-reduced.** `beta` is pi-periodic, so the box covers exactly one
  period rather than wrapping over 32 copies of every optimum (see `angles.canonicalise`).
- **The objective is log P(ground).** P spans four orders of magnitude, from 1/4096 at the
  uniform state to ~0.3; a GP on raw P models only the top of that range.
- **Gradient polish closes it out.** BO chooses the basin, Adam descends it. Neither is good
  at the other's job, and we have exact gradients, so refusing to use them would be an
  affectation.
- **Matched-budget baselines run alongside.** TQA+Adam and random+Adam at the same number of
  circuit evaluations, because "TuRBO scored X" means nothing without them.

Usage:
    python -m src.turbo --evals 400 --out runs/turbo_submission.csv
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from .angles import DEPTH, angle_scale, to_angles
from .predict import polish, write_submission
from .qaoa_ref import QAOA
from .schedules import sample_starts, tqa_units

N_ANGLES = 2 * DEPTH
GAMMA_BOX = 3.0          # search box half-width for gamma, in normalised units
BETA_BOX = 1.0           # exactly one mixer period -- no point searching further
SQRT5 = math.sqrt(5.0)


# ----------------------------------------------------------------------------------
# batched exact GP: Matern 5/2 with ARD
# ----------------------------------------------------------------------------------

def matern52(A, B, ls):
    """(b,n,d), (b,m,d), (b,d) -> (b,n,m). Unit output scale."""
    r = torch.cdist(A / ls.unsqueeze(1), B / ls.unsqueeze(1)).clamp_min(0)
    return (1 + SQRT5 * r + (5.0 / 3.0) * r * r) * torch.exp(-SQRT5 * r)


def gp_factor(X, y, valid, ls, outscale, noise):
    """Cholesky of the training covariance, and alpha = K^-1 y.

    Invalid buffer slots are neutralised rather than removed, which keeps every instance
    the same shape and so keeps the whole step batched: their rows and columns are zeroed
    and their diagonal set to 1, making K block-diagonal (real block + identity). The
    Cholesky inherits that structure, so those slots contribute exactly nothing to the
    posterior while the tensor stays rectangular.
    """
    b, n, _ = X.shape
    vv = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    K = outscale.view(-1, 1, 1) * matern52(X, X, ls) * vv
    diag = torch.where(valid, noise.view(-1, 1).expand(b, n), torch.ones_like(K[:, 0, :]))
    K = K + torch.diag_embed(diag)
    # Thompson sampling can propose a point twice, which makes K singular. Escalating
    # jitter is cheaper and safer than losing a GPU-hour run to a LinAlgError.
    for jitter in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = torch.linalg.cholesky(
                K + jitter * torch.eye(n, device=K.device, dtype=K.dtype))
            break
        except Exception:
            if jitter == 1e-2:
                raise
    alpha = torch.cholesky_solve((y * valid).unsqueeze(-1), L)
    return L, alpha


def gp_predict(Xc, X, L, alpha, valid, ls, outscale):
    """Posterior mean and variance at candidates Xc (b, m, d)."""
    Ks = outscale.view(-1, 1, 1) * matern52(Xc, X, ls) * valid.unsqueeze(1)
    mean = (Ks @ alpha).squeeze(-1)
    v = torch.linalg.solve_triangular(L, Ks.transpose(-1, -2), upper=False)
    var = outscale.view(-1, 1) - (v * v).sum(dim=-2)
    return mean, var.clamp_min(1e-10)


def fit_hypers(X, y, valid, ls, outscale, noise, steps=30, lr=0.08, sub=96, gen=None):
    """Shared hyperparameters by maximising the summed log marginal likelihood.

    Per-instance hyperparameters would be the textbook choice, but every instance here is
    drawn from one ensemble with one fixed J, so the lengthscales genuinely are shared —
    fitting them jointly is both cheaper and better conditioned than fitting 500 GPs on a
    hundred points each. A random subset of instances is used per refit to keep the cost
    negligible against the circuit evaluations.
    """
    b = X.shape[0]
    idx = torch.randperm(b, generator=gen, device=X.device)[:min(sub, b)]
    Xs, ys, vs = X[idx], y[idx], valid[idx]
    p_ls = ls.log().clone().requires_grad_(True)
    p_os = outscale.log().clone().requires_grad_(True)
    p_no = noise.log().clone().requires_grad_(True)
    opt = torch.optim.Adam([p_ls, p_os, p_no], lr=lr)
    n_eff = vs.sum(dim=1).clamp_min(1).double()
    for _ in range(steps):
        opt.zero_grad()
        lsv = p_ls.exp().clamp(1e-2, 10.0).expand(len(idx), -1)
        osv = p_os.exp().clamp(1e-3, 1e3).expand(len(idx))
        nov = p_no.exp().clamp(1e-6, 1.0).expand(len(idx))
        L, alpha = gp_factor(Xs, ys, vs, lsv, osv, nov)
        fit = 0.5 * ((ys * vs).unsqueeze(-1) * alpha).sum(dim=(1, 2))
        cplx = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1)
        (-(-fit - cplx - 0.5 * n_eff * math.log(2 * math.pi)).mean()).backward()
        opt.step()
    with torch.no_grad():
        return (p_ls.exp().clamp(1e-2, 10.0).detach(),
                p_os.exp().clamp(1e-3, 1e3).detach(),
                p_no.exp().clamp(1e-6, 1.0).detach())


# ----------------------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------------------

def box(device):
    lo = torch.tensor([-GAMMA_BOX] * DEPTH + [-BETA_BOX] * DEPTH, device=device)
    return lo, -lo


def to_unit(u, lo, hi):
    return (u - lo) / (hi - lo)


def from_unit(x, lo, hi):
    return lo + x * (hi - lo)


class Turbo:
    """N x n_tr trust regions advanced in lockstep."""

    def __init__(self, n, d, device, gen, n_tr=2, l_init=0.8, l_min=0.5 ** 7,
                 l_max=1.6, tau_succ=3, tau_fail=8, mem=128):
        self.b, self.d, self.n, self.n_tr = n * n_tr, d, n, n_tr
        self.device, self.gen, self.mem = device, gen, mem
        self.l_init, self.l_min, self.l_max = l_init, l_min, l_max
        self.tau_succ, self.tau_fail = tau_succ, tau_fail
        self.L = torch.full((self.b,), l_init, device=device, dtype=torch.float64)
        self.succ = torch.zeros(self.b, device=device)
        self.fail = torch.zeros(self.b, device=device)
        self.X = torch.zeros(self.b, mem, d, device=device, dtype=torch.float64)
        self.y = torch.zeros(self.b, mem, device=device, dtype=torch.float64)
        self.valid = torch.zeros(self.b, mem, dtype=torch.bool, device=device)
        self.ptr = torch.zeros(self.b, dtype=torch.long, device=device)
        self.restarts = 0

    def add(self, xs, ys, idx=None):
        """Append a batch of (candidate, value) into each trust region's ring buffer."""
        rows = torch.arange(self.b, device=self.device) if idx is None else idx
        for j in range(xs.shape[1]):
            p = self.ptr[rows] % self.mem
            self.X[rows, p] = xs[:, j]
            self.y[rows, p] = ys[:, j]
            self.valid[rows, p] = True
            self.ptr[rows] += 1

    def best(self):
        yy = torch.where(self.valid, self.y, torch.full_like(self.y, -1e30))
        v, i = yy.max(dim=1)
        return self.X[torch.arange(self.b, device=self.device), i], v

    def update(self, gained):
        """TuRBO's success/failure counters: streaks resize the region."""
        self.succ = torch.where(gained, self.succ + 1, torch.zeros_like(self.succ))
        self.fail = torch.where(gained, torch.zeros_like(self.fail), self.fail + 1)
        up = self.succ >= self.tau_succ
        dn = self.fail >= self.tau_fail
        self.L = torch.where(up, (self.L * 2).clamp(max=self.l_max), self.L)
        self.L = torch.where(dn, self.L / 2, self.L)
        self.succ = torch.where(up | dn, torch.zeros_like(self.succ), self.succ)
        self.fail = torch.where(up | dn, torch.zeros_like(self.fail), self.fail)
        return self.L < self.l_min                      # regions that have collapsed

    def reseed(self, mask):
        """A collapsed region has converged; drop its history and restart it elsewhere."""
        if not mask.any():
            return
        self.L[mask] = self.l_init
        self.succ[mask] = 0
        self.fail[mask] = 0
        self.valid[mask] = False
        self.ptr[mask] = 0
        self.restarts += int(mask.sum())

    def candidates(self, ls, n_cand):
        """Perturb each region's incumbent inside a box scaled by the GP lengthscales.

        Anisotropic scaling is the part of TuRBO that matters most here: gamma and beta
        have natural scales an order of magnitude apart, so an isotropic region would be
        simultaneously too coarse for one half of the vector and too fine for the other.
        """
        c, _ = self.best()
        w = ls / ls.prod(dim=1, keepdim=True).pow(1.0 / self.d)
        side = (self.L.unsqueeze(1) * w).clamp(1e-4, 1.0)
        lo = (c - side / 2).clamp(0, 1)
        hi = (c + side / 2).clamp(0, 1)
        r = torch.rand(self.b, n_cand, self.d, device=self.device,
                       dtype=torch.float64, generator=self.gen)
        return lo.unsqueeze(1) + r * (hi - lo).unsqueeze(1)


def make_objective(sim, h, scale, n_tr, chunk=8192):
    """log P(ground), batched over (trust region, instance) and evaluated in chunks."""
    h_rep = h.repeat(n_tr, 1)
    lo, hi = box(h.device)
    counter = {"evals": 0}

    @torch.no_grad()
    def f(x_unit, idx=None):                         # (b, m, d) in [0,1] -> (b, m)
        b, m, d = x_unit.shape
        u = from_unit(x_unit.reshape(-1, d).float(), lo, hi)
        hh = (h_rep if idx is None else h_rep[idx]).repeat_interleave(m, dim=0)
        out = []
        for s in range(0, u.shape[0], chunk):
            a = to_angles(u[s:s + chunk], scale)
            p = sim.p_ground(hh[s:s + chunk], a[:, :DEPTH], a[:, DEPTH:])
            out.append(torch.log(p.clamp_min(1e-12)))
        counter["evals"] += u.shape[0]
        return torch.cat(out).view(b, m).double()

    return f, counter, (lo, hi)


def run(sim, h, scale, args, gen):
    dev = h.device
    n = h.shape[0]
    f, counter, (lo, hi) = make_objective(sim, h, scale, args.n_tr, args.chunk)
    tb = Turbo(n, N_ANGLES, dev, gen, n_tr=args.n_tr, mem=args.mem,
               tau_fail=args.tau_fail)

    def seed(mask=None):
        """Initial design: half Sobol, half annealing-schedule points (see schedules.py).

        Only the regions being seeded are evaluated. Seeding all of them on every restart
        would spend the whole budget re-measuring regions that never restarted.
        """
        idx = (torch.arange(tb.b, device=dev) if mask is None
               else mask.nonzero(as_tuple=True)[0])
        if idx.numel() == 0:
            return
        k, m = idx.numel(), args.n_init
        # SobolEngine's seed is a plain int; the RNG here lives on the compute device, so
        # draw the scramble seed on that device rather than implicitly on the CPU.
        sd = int(torch.randint(1 << 30, (1,), device=dev, generator=gen).item())
        sob = torch.quasirandom.SobolEngine(N_ANGLES, scramble=True, seed=sd)
        xs = sob.draw(k * m).to(dev).double().view(k, m, N_ANGLES)
        n_sched = m // 2
        u = sample_starts(k * n_sched, "ramp", dev, gen)
        xs[:, :n_sched] = to_unit(u.double(), lo, hi).view(k, n_sched, N_ANGLES).clamp(0, 1)
        tb.add(xs, f(xs, idx), idx)

    seed()
    ls = torch.ones(tb.b, N_ANGLES, device=dev, dtype=torch.float64) * 0.3
    osc = torch.ones(tb.b, device=dev, dtype=torch.float64)
    nsc = torch.full((tb.b,), 1e-3, device=dev, dtype=torch.float64)

    t0, it = time.time(), 0
    budget = args.evals * n
    while counter["evals"] < budget:
        it += 1
        if it % args.hp_every == 1 or it == 1:
            l1, o1, n1 = fit_hypers(tb.X, standardise(tb.y, tb.valid), tb.valid,
                                    ls[:1].squeeze(0), osc[:1], nsc[:1],
                                    steps=args.hp_steps, sub=args.hp_sub, gen=gen)
            ls, osc, nsc = l1.expand(tb.b, -1), o1.expand(tb.b), n1.expand(tb.b)

        ystd = standardise(tb.y, tb.valid)
        L, alpha = gp_factor(tb.X, ystd, tb.valid, ls, osc, nsc)
        xc = tb.candidates(ls, args.n_cand)
        mu, var = gp_predict(xc, tb.X, L, alpha, tb.valid, ls, osc)
        # Thompson sampling on independent marginals -- the joint draw TuRBO specifies
        # needs an (n_cand x n_cand) Cholesky per region, which does not batch at this size
        draw = mu + var.sqrt() * torch.randn(mu.shape, device=dev, dtype=torch.float64,
                                             generator=gen)
        pick = draw.topk(args.batch, dim=1).indices
        xs = torch.gather(xc, 1, pick.unsqueeze(-1).expand(-1, -1, N_ANGLES))
        ys = f(xs)

        _, prev = tb.best()
        tb.add(xs, ys)
        _, now = tb.best()
        dead = tb.update(now > prev + 1e-3 * prev.abs())
        if dead.any():
            tb.reseed(dead)
            seed(dead)

        if it % args.log_every == 0:
            xb, yb = tb.best()
            pb = yb.view(args.n_tr, n).max(dim=0).values.exp()
            print(f"  it {it:4d}  evals/inst {counter['evals'] // n:6d}  "
                  f"mean P {pb.mean():.4f}  median L {tb.L.median():.4f}  "
                  f"restarts {tb.restarts}  {time.time() - t0:5.0f}s", flush=True)

    xb, yb = tb.best()
    xb = xb.view(args.n_tr, n, N_ANGLES)
    best_tr = yb.view(args.n_tr, n).argmax(dim=0)
    u = from_unit(xb[best_tr, torch.arange(n, device=dev)].float(), lo, hi)
    return u, counter["evals"] // n, tb.restarts


def standardise(y, valid):
    cnt = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mu = (y * valid).sum(dim=1, keepdim=True) / cnt
    sd = (((y - mu) * valid) ** 2).sum(dim=1, keepdim=True).div(cnt).sqrt().clamp_min(1e-6)
    return (y - mu) / sd


@torch.no_grad()
def score(sim, h, u, scale):
    a = to_angles(u, scale)
    return sim.p_ground(h, a[:, :DEPTH], a[:, DEPTH:])


def baseline(sim, h, scale, kind, n_starts, polish_steps, gen, chunk):
    """Matched-budget control: n_starts from `kind`, Adam on each, keep the best."""
    n = h.shape[0]
    best = torch.zeros(n, device=h.device)
    for _ in range(n_starts):
        u0 = sample_starts(n, kind, h.device, gen)
        u = polish(sim, h, u0, scale, polish_steps, chunk=chunk)
        best = torch.maximum(best, score(sim, h, u, scale))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--h", type=Path, default=None, help="default: data-dir/h_train.npy")
    ap.add_argument("--out", type=Path, default=Path("runs/turbo_submission.csv"))
    ap.add_argument("--evals", type=int, default=400, help="circuit evals per instance")
    ap.add_argument("--n-tr", type=int, default=2, help="trust regions per instance")
    ap.add_argument("--n-init", type=int, default=20, help="initial design per region")
    ap.add_argument("--n-cand", type=int, default=192, help="Thompson candidates per step")
    ap.add_argument("--batch", type=int, default=4, help="picks per region per step")
    ap.add_argument("--mem", type=int, default=128, help="GP memory per region")
    ap.add_argument("--tau-fail", type=int, default=8)
    ap.add_argument("--hp-every", type=int, default=10)
    ap.add_argument("--hp-steps", type=int, default=25)
    ap.add_argument("--hp-sub", type=int, default=96)
    ap.add_argument("--polish", type=int, default=200, help="final Adam steps")
    ap.add_argument("--baseline-starts", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    hp = args.h or (args.data_dir / "h_train.npy")
    h = torch.tensor(np.load(hp), dtype=torch.float32, device=dev)
    scale = angle_scale(sim, torch.tensor(np.load(args.data_dir / "h_train.npy"),
                                          dtype=torch.float32, device=dev), device=dev)

    print(f"TuRBO on {h.shape[0]} instances | {args.n_tr} trust regions each | "
          f"{args.evals} circuit evals per instance | {dev}", flush=True)
    t0 = time.time()
    u, used, restarts = run(sim, h, scale, args, gen)
    p_bo = score(sim, h, u, scale)
    t_bo = time.time() - t0
    print(f"\nTuRBO alone            : mean P {p_bo.mean():.5f}  median {p_bo.median():.5f}"
          f"   ({used} evals/inst, {restarts} restarts, {t_bo:.0f}s)", flush=True)

    u = polish(sim, h, u, scale, args.polish, chunk=args.chunk)
    p_hy = score(sim, h, u, scale)
    t_all = time.time() - t0
    print(f"TuRBO + {args.polish} Adam steps : mean P {p_hy.mean():.5f}  "
          f"median {p_hy.median():.5f}   ({t_all:.0f}s)", flush=True)

    for kind in ("tqa", "uniform"):
        pb = baseline(sim, h, scale, kind, args.baseline_starts, args.polish, gen, args.chunk)
        print(f"{kind:>7} x{args.baseline_starts} + Adam    : mean P {pb.mean():.5f}  "
              f"median {pb.median():.5f}", flush=True)

    budget = "within" if t_all < 600 else "OVER"
    print(f"\ntotal {t_all:.0f}s — {budget} the 10-minute inference budget")
    print("reference: proposer 0.18405 | multistart nb 03 0.32396 | leaderboard #1 0.81468")
    write_submission(args.out, to_angles(u, scale))
    return {"submission": args.out, "mean_p": p_hy.mean().item(),
            "mean_p_no_polish": p_bo.mean().item(), "seconds": t_all,
            "evals_per_instance": used, "restarts": restarts}


if __name__ == "__main__":
    main()
