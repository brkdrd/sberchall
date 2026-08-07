import torch

from .model import N_ANGLES
from .qaoa_ref import P as DEPTH

P_FLOOR = 1e-12        # P(ground) > 0 always, but guard the log anyway
GRAD_CLIP = 10.0       # gradient *features* are clipped to a sane input range


def log_p(sim, h, angles):
    """Differentiable log P(ground) for a batch of (h, angles) rows."""
    gamma, beta = angles[:, :DEPTH], angles[:, DEPTH:]
    return torch.log(sim.p_ground(h, gamma, beta).clamp_min(P_FLOOR))


def eval_token(sim, h, angles):
    """Evaluate a detached point: returns the 21-dim token and logP.

    The token packs (angles, logP, d logP / d angles). Everything is detached —
    these are observations fed back to the model, not part of the training graph.
    """
    a = angles.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = log_p(sim, h, a)
        (g,) = torch.autograd.grad(lp.sum(), a)
    token = torch.cat(
        [a.detach(), lp.detach().unsqueeze(1), g.clamp(-GRAD_CLIP, GRAD_CLIP)], dim=1
    )
    return token, lp.detach()


def init_angles(batch, device):
    return (torch.rand(batch, N_ANGLES, device=device) - 0.5) * torch.pi


def rollout(model, sim, h, n_steps, noise_std=0.0, train=True):
    """Autoregressive rollout with the simulator in the loop.

    History tokens are detached (truncated BPTT): the training gradient flows
    from each step's -logP loss through the simulator into that step's
    prediction only. Exploration noise is reparameterised, so it does not cut
    the gradient path.

    Returns (losses, traj_angles, traj_logp):
      losses       list of n_steps differentiable (B,) tensors, or [] if not train
      traj_angles  (B, n_steps + 1, 10) detached — includes the random start
      traj_logp    (B, n_steps + 1) detached
    """
    token, lp = eval_token(sim, h, init_angles(h.shape[0], h.device))
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
            losses.append(-log_p(sim, h, pred))
        token, lp = eval_token(sim, h, pred)
        tokens.append(token)
        traj_logp.append(lp)
    traj_angles = torch.stack([t[:, :N_ANGLES] for t in tokens], dim=1)
    return losses, traj_angles, torch.stack(traj_logp, dim=1)


def best_of_rollouts(model, sim, h, n_steps, n_restarts, chunk=512):
    """Run `n_restarts` independent rollouts per instance, return the best
    angles (over all restarts and all trajectory steps) and their P(ground).

    h: (N, 12) tensor. Returns (angles (N, 10), p (N,)).
    """
    n = h.shape[0]
    device = h.device
    best_p = torch.zeros(n, device=device)
    best_angles = torch.zeros(n, N_ANGLES, device=device)
    for r in range(n_restarts):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            _, ang, lp = rollout(model, sim, h[lo:hi], n_steps, noise_std=0.0, train=False)
            p = lp.exp()                                  # (b, T+1)
            step_best, step_idx = p.max(dim=1)            # best step of this rollout
            better = step_best > best_p[lo:hi]
            rows = torch.nonzero(better, as_tuple=True)[0]
            best_p[lo:hi][rows] = step_best[rows]
            best_angles[lo:hi][rows] = ang[rows, step_idx[rows]]
    return best_angles, best_p
