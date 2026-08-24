import torch

from .angles import to_angles
from .model import N_ANGLES
from .qaoa_ref import P as DEPTH

P_FLOOR = 1e-12        # P(ground) > 0 always, but guard the log anyway
GRAD_CLIP = 10.0       # gradient *features* are clipped to a sane input range


def log_p(sim, h, u, scale):
    """Differentiable log P(ground) for a batch of (h, normalised-angle) rows."""
    a = to_angles(u, scale)
    return torch.log(sim.p_ground(h, a[:, :DEPTH], a[:, DEPTH:]).clamp_min(P_FLOOR))


def eval_token(sim, h, u, scale):
    """Evaluate a detached point: returns the 21-dim token and logP.

    The token packs (units, logP, d logP / d units). Everything is detached — these are
    observations fed back to the model, not part of the training graph. The gradient is
    taken w.r.t. the *normalised* coordinates, so the chain rule folds `scale` in and the
    gamma and beta halves arrive on a common magnitude instead of the gamma half
    saturating GRAD_CLIP.
    """
    uu = u.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = log_p(sim, h, uu, scale)
        (g,) = torch.autograd.grad(lp.sum(), uu)
    token = torch.cat(
        [uu.detach(), lp.detach().unsqueeze(1), g.clamp(-GRAD_CLIP, GRAD_CLIP)], dim=1
    )
    return token, lp.detach()


def init_units(batch, device):
    """Uniform over the normalised box: one phase revolution in gamma, one period in beta."""
    return torch.rand(batch, N_ANGLES, device=device) * 2 - 1


def rollout(model, sim, h, n_steps, scale, noise_std=0.0, train=True):
    """Autoregressive rollout with the simulator in the loop.

    Everything here is in normalised units (see `angles.py`); `scale` converts to radians
    at the simulator boundary only. That keeps one exploration `noise_std` and one model
    output scale meaningful for both gamma and beta.

    History tokens are detached (truncated BPTT): the training gradient flows
    from each step's -logP loss through the simulator into that step's
    prediction only. Exploration noise is reparameterised, so it does not cut
    the gradient path.

    Returns (losses, traj_units, traj_logp):
      losses      list of n_steps differentiable (B,) tensors, or [] if not train
      traj_units  (B, n_steps + 1, 10) detached, normalised — includes the random start
      traj_logp   (B, n_steps + 1) detached
    """
    token, lp = eval_token(sim, h, init_units(h.shape[0], h.device), scale)
    tokens, traj_logp = [token], [lp]
    losses = []
    for _ in range(n_steps):
        seq = torch.stack(tokens, dim=1)
        if train:
            delta = model(seq, h)
        else:
            with torch.no_grad():
                delta = model(seq, h)
        pred = seq[:, -1, :N_ANGLES] + delta
        if noise_std > 0:
            pred = pred + noise_std * torch.randn_like(pred)
        if train:
            losses.append(-log_p(sim, h, pred, scale))
        token, lp = eval_token(sim, h, pred, scale)
        tokens.append(token)
        traj_logp.append(lp)
    traj_units = torch.stack([t[:, :N_ANGLES] for t in tokens], dim=1)
    return losses, traj_units, torch.stack(traj_logp, dim=1)


def restart_candidates(model, sim, h, n_steps, n_restarts, scale, chunk=512):
    """Best trajectory point of every restart, kept separately per restart.

    h: (N, 12). Returns (units (R, N, 10), p (R, N)) — R candidate points per
    instance, in normalised units, for polish-then-select at inference.
    """
    n = h.shape[0]
    all_a, all_p = [], []
    for r in range(n_restarts):
        pa, aa = [], []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            _, u, lp = rollout(model, sim, h[lo:hi], n_steps, scale, noise_std=0.0, train=False)
            p = lp.exp()
            step_best, step_idx = p.max(dim=1)
            pa.append(step_best)
            aa.append(u[torch.arange(hi - lo, device=h.device), step_idx])
        all_p.append(torch.cat(pa))
        all_a.append(torch.cat(aa))
    return torch.stack(all_a), torch.stack(all_p)


def best_of_rollouts(model, sim, h, n_steps, n_restarts, scale, chunk=512):
    """Run `n_restarts` independent rollouts per instance, return the best
    units (over all restarts and all trajectory steps) and their P(ground).

    h: (N, 12) tensor. Returns (units (N, 10), p (N,)) — normalised, not radians.
    """
    n = h.shape[0]
    device = h.device
    best_p = torch.zeros(n, device=device)
    best_units = torch.zeros(n, N_ANGLES, device=device)
    for r in range(n_restarts):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            _, u, lp = rollout(model, sim, h[lo:hi], n_steps, scale, noise_std=0.0, train=False)
            p = lp.exp()                                  # (b, T+1)
            step_best, step_idx = p.max(dim=1)            # best step of this rollout
            better = step_best > best_p[lo:hi]
            rows = torch.nonzero(better, as_tuple=True)[0]
            best_p[lo:hi][rows] = step_best[rows]
            best_units[lo:hi][rows] = u[rows, step_idx[rows]]
    return best_units, best_p
