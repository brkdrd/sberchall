"""Evaluate and locally refine a batch of angle vectors. Shared by every searcher here.

Two primitives, both taking angles in **radians** and going straight through the
organisers' simulator so nothing of ours sits between a point and its score:

- `score`   — P(ground) for a batch, forward only, chunked.
- `refine`  — Adam on the angles, returning the best point *visited* rather than the last
              iterate, because P(ground) is not monotone along an Adam path.

`refine` keeps separate step sizes for the two halves of the angle vector. That is not a
reparameterisation of the search space — it is the observation that a step meaningful for
beta (period pi) is invisible to a gamma of order 10, so one step size cannot serve both.
"""

import torch

from .qaoa_ref import P as DEPTH

N_ANGLES = 2 * DEPTH
P_FLOOR = 1e-30


@torch.no_grad()
def score(sim, h, ang, chunk=4096):
    """P(ground) for a batch of (h, angles) rows, via the organisers' simulator."""
    out = []
    for lo in range(0, ang.shape[0], chunk):
        a, hh = ang[lo:lo + chunk], h[lo:lo + chunk]
        out.append(sim.p_ground(hh, a[:, :DEPTH], a[:, DEPTH:]))
    return torch.cat(out)


def refine(sim, h, ang, steps, lr_gamma, lr_beta, chunk=1024, p0=None):
    """Adam on the angles. Returns (best point visited, its P(ground)).

    `p0` skips the initial scoring pass when the caller already has it. The chunk size is
    a VRAM knob: autograd through the five layers stores ~60 intermediates of
    (chunk, 4096) complex64, i.e. about 2 MB per row.
    """
    best_ang = ang.clone()
    best_p = score(sim, h, ang, chunk * 4) if p0 is None else p0.clone()
    if steps <= 0:
        return best_ang, best_p
    # this function descends a gradient internally, so it must not inherit the caller's
    # grad mode — it is called from inside no_grad evaluation paths
    with torch.enable_grad():
        for lo in range(0, ang.shape[0], chunk):
            sl = slice(lo, min(lo + chunk, ang.shape[0]))
            g = ang[sl, :DEPTH].detach().clone().requires_grad_(True)
            b = ang[sl, DEPTH:].detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([{"params": [g], "lr": lr_gamma},
                                    {"params": [b], "lr": lr_beta}])
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                p = sim.p_ground(h[sl], g, b)
                (-p.clamp_min(P_FLOOR).log()).sum().backward()
                opt.step()
                with torch.no_grad():
                    # Branchless on purpose. `if better.any()` and `torch.nonzero` both
                    # force a GPU->CPU synchronisation, and this runs once per Adam step
                    # per chunk — so the pipeline was drained thousands of times a second
                    # and never got to queue any work. torch.where costs one extra kernel
                    # and no stall.
                    pd = p.detach()
                    better = pd > best_p[sl]
                    best_p[sl] = torch.where(better, pd, best_p[sl])
                    best_ang[sl] = torch.where(better.unsqueeze(1),
                                               torch.cat([g, b], dim=1).detach(),
                                               best_ang[sl])
    return best_ang, best_p
