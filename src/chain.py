"""The search this policy drives: a chain of nodes, each one surrounded by its own probes.

A **node** is a point in angle space. Its **surroundings** are `k` points sampled in a
ball around it, each evaluated, then each run forward under Adam for a fixed number of
steps. The node owns all of it: where each probe started, where it finished, and what
P(ground) was at both ends. That is the observation — not a single number saying how good
the node is, but a picture of what the landscape does around it and which way the local
flow runs.

The cycle:

    root <- RootMLP(h), explored
    repeat:
        child 1 <- head + policy(head's surroundings)
        child 2 <- head + policy(head's surroundings, child 1's)
        child 3 <- head + policy(head's surroundings, child 1's, child 2's)
        head, parent, grandparent <- child 3, head, parent

So the first two children are probes the policy places deliberately and then gets to read
before committing; the third is the commitment. The policy sees three generations back,
which is what lets it recognise a direction it has already tried.

Two properties make this trainable end to end with nothing but REINFORCE:

- every node is `parent centre + emitted offset`, so a chain is a sequence of actions with
  a shared terminal reward and no hidden state the gradient has to cross;
- the probes and their Adam finishes are **observations**, detached on the way in. They are
  where the compute goes, but no gradient flows through them — which is the point, since
  the map `start -> Adam finish` is piecewise constant in value, so a pathwise gradient
  through it is zero wherever it exists and undefined at the basin boundaries where it
  does not. Which basin a node lands in is a discrete event, and the score-function
  estimator is the one that can see it.

Cost is exact and worth keeping in mind: a node costs `k * (1 + 2 * adam_steps)` forward
passes, and a chain has `1 + 3 * iters` nodes.
"""

import math
from dataclasses import dataclass, field

import torch

from .policy import CHILD1, CHILD2, GRAND, HEAD, N_ANGLES, PARENT, TOKEN_DIM
from .qaoa_ref import P as DEPTH
from .refine import P_FLOOR, refine, score

N_CHILDREN = 3
LP_SCALE = 10.0        # log P spans about -9 .. 0; this puts tokens near unit scale


def vec(g, b, device):
    """A per-coordinate vector from one value for gamma and one for beta."""
    return torch.tensor([g] * DEPTH + [b] * DEPTH, dtype=torch.float32, device=device)


@dataclass
class ChainConfig:
    """Every knob. The two halves of the angle vector get separate values throughout —
    not a reparameterisation, just the fact that a step meaningful for beta (period pi)
    is invisible to a gamma of order 10.

    The exploration scales are measured, not chosen: `reinforce.probe_jitter` moves the
    root schedule by a ladder of sigmas and takes the widest that still holds quality.
    These defaults are what it returned; the first run shipped with gamma noise at 0.6
    against a schedule whose mean |gamma| is 0.30, so every chain opened by discarding the
    one thing known to work."""

    k: int = 6                     # probes per node
    iters: int = 4                 # head advances per chain
    radius_gamma: float = 0.05     # ball the probes are drawn from
    radius_beta: float = 0.10
    adam_steps: int = 12           # Adam steps run from each probe
    lr_gamma: float = 0.05
    lr_beta: float = 0.03
    sigma_gamma: float = 0.05      # exploration noise on the emitted offset
    sigma_beta: float = 0.10
    sigma_root_gamma: float = 0.10  # and on the root MLP's point
    sigma_root_beta: float = 0.20
    pos_scale_gamma: float = 1.0   # token input normalisation only
    pos_scale_beta: float = 1.0
    chunk: int = 512               # rows refined at once (VRAM: ~2 MB/row)
    score_chunk: int = 8192

    def tensors(self, device):
        return {
            "radius": vec(self.radius_gamma, self.radius_beta, device),
            "sigma": vec(self.sigma_gamma, self.sigma_beta, device),
            "sigma_root": vec(self.sigma_root_gamma, self.sigma_root_beta, device),
            "pos_scale": vec(self.pos_scale_gamma, self.pos_scale_beta, device),
        }

    def evals_per_node(self):
        return self.k * (1 + 2 * self.adam_steps)

    def evals_per_chain(self):
        return (1 + N_CHILDREN * self.iters) * self.evals_per_node()


@dataclass
class Node:
    centre: torch.Tensor       # (N, 10)
    starts: torch.Tensor       # (N, k, 10)
    fins: torch.Tensor         # (N, k, 10)
    lp_start: torch.Tensor     # (N, k)
    lp_fin: torch.Tensor       # (N, k)
    best_lp: torch.Tensor      # (N,)
    best_pt: torch.Tensor      # (N, 10)


def ball(n, k, dim, gen, device):
    """`k` points uniform in the unit ball of R^dim, the first of which is the centre.

    Keeping probe 0 at the centre means a node's own value is always in its surroundings,
    so the policy never has to infer it from neighbours.
    """
    g = torch.randn(n, k, dim, device=device, generator=gen)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = torch.rand(n, k, 1, device=device, generator=gen) ** (1.0 / dim)
    u = g * r
    u[:, 0] = 0.0
    return u


def explore(sim, h, centre, cfg, tens, gen):
    """Sample a ball around `centre`, evaluate it, and run Adam from every probe."""
    centre = centre.detach()
    n, k = centre.shape[0], cfg.k
    u = ball(n, k, N_ANGLES, gen, centre.device)
    starts = centre.unsqueeze(1) + u * tens["radius"]
    flat = starts.reshape(n * k, N_ANGLES)
    hrep = h.repeat_interleave(k, dim=0)

    p_start = score(sim, hrep, flat, cfg.score_chunk)
    fins, p_fin = refine(sim, hrep, flat, cfg.adam_steps, cfg.lr_gamma, cfg.lr_beta,
                         cfg.chunk, p0=p_start)

    lp_start = p_start.clamp_min(P_FLOOR).log().view(n, k)
    lp_fin = p_fin.clamp_min(P_FLOOR).log().view(n, k)
    fins = fins.view(n, k, N_ANGLES)

    allp = torch.cat([lp_start, lp_fin], dim=1)
    allpts = torch.cat([starts, fins], dim=1)
    idx = allp.argmax(dim=1)
    rows = torch.arange(n, device=centre.device)
    return Node(centre=centre, starts=starts, fins=fins, lp_start=lp_start,
                lp_fin=lp_fin, best_lp=allp[rows, idx], best_pt=allpts[rows, idx])


def node_tokens(node, head_centre, pos_scale):
    """One token per probe, every position relative to the head's centre."""
    hc = head_centre.unsqueeze(1)
    start_rel = (node.starts - hc) / pos_scale
    fin_rel = (node.fins - hc) / pos_scale
    disp = (node.fins - node.starts) / pos_scale
    node_rel = ((node.centre - head_centre) / pos_scale).unsqueeze(1).expand_as(start_rel)
    lp_s = (node.lp_start / LP_SCALE).unsqueeze(-1)
    lp_f = (node.lp_fin / LP_SCALE).unsqueeze(-1)
    return torch.cat([start_rel, fin_rel, disp, node_rel, lp_s, lp_f, lp_f - lp_s], dim=-1)


def context(head, parent, grand, children, cfg, tens):
    """Pack the surroundings of up to five nodes into (tokens, roles, valid)."""
    slots = [(HEAD, head), (PARENT, parent), (GRAND, grand),
             (CHILD1, children[0] if len(children) > 0 else None),
             (CHILD2, children[1] if len(children) > 1 else None)]
    n, k, dev = head.centre.shape[0], cfg.k, head.centre.device
    toks, roles, valid = [], [], []
    for role, node in slots:
        if node is None:
            toks.append(torch.zeros(n, k, TOKEN_DIM, device=dev))
            valid.append(torch.zeros(n, k, dtype=torch.bool, device=dev))
        else:
            toks.append(node_tokens(node, head.centre, tens["pos_scale"]))
            valid.append(torch.ones(n, k, dtype=torch.bool, device=dev))
        roles.append(torch.full((n, k), role, dtype=torch.long, device=dev))
    return (torch.cat(toks, dim=1).detach(), torch.cat(roles, dim=1),
            torch.cat(valid, dim=1))


def cond_vector(h, head, it, child_idx, cfg, tens):
    """h, plus where the head actually is and how far through the chain we are."""
    depth = torch.full((h.shape[0], 1), it / max(cfg.iters, 1), device=h.device)
    child = torch.full((h.shape[0], 1), child_idx / N_CHILDREN, device=h.device)
    return torch.cat([h, head.centre / tens["pos_scale"],
                      (head.best_lp / LP_SCALE).unsqueeze(-1), depth, child], dim=1).detach()


def gauss_logp(action, mu, sigma):
    """log N(action; mu, diag(sigma^2)), summed over the ten coordinates."""
    z = (action - mu) / sigma
    return (-0.5 * z ** 2 - sigma.log() - 0.5 * math.log(2 * math.pi)).sum(-1)


def rollout(root_mlp, policy, sim, h, cfg, gen, explore_scale=1.0):
    """One chain per row of `h`. Returns the terminal reward and the chain's log-prob.

    `explore_scale` multiplies the policy noise; 0 makes the chain deterministic, which is
    what a "what does the mean policy do" diagnostic wants and not what inference wants.
    """
    dev = h.device
    tens = cfg.tensors(dev)
    n = h.shape[0]
    sigma = tens["sigma"] * explore_scale
    sigma_root = tens["sigma_root"] * explore_scale

    mu = root_mlp(h)
    if explore_scale > 0:
        action = (mu + sigma_root * torch.randn(n, N_ANGLES, device=dev, generator=gen))
        logps = [gauss_logp(action.detach(), mu, sigma_root)]
    else:
        action, logps = mu, [torch.zeros(n, device=dev)]
    head = explore(sim, h, action.detach(), cfg, tens, gen)

    best_lp, best_pt = head.best_lp.clone(), head.best_pt.clone()
    parent = grand = None

    for it in range(cfg.iters):
        children = []
        for j in range(N_CHILDREN):
            tokens, roles, valid = context(head, parent, grand, children, cfg, tens)
            mu = policy(tokens, roles, valid, cond_vector(h, head, it, j, cfg, tens))
            if explore_scale > 0:
                a = mu + sigma * torch.randn(n, N_ANGLES, device=dev, generator=gen)
                logps.append(gauss_logp(a.detach(), mu, sigma))
            else:
                a = mu
                logps.append(torch.zeros(n, device=dev))
            child = explore(sim, h, head.centre + a.detach(), cfg, tens, gen)
            children.append(child)
            better = child.best_lp > best_lp
            best_lp = torch.where(better, child.best_lp, best_lp)
            best_pt = torch.where(better.unsqueeze(-1), child.best_pt, best_pt)
        grand, parent, head = parent, head, children[N_CHILDREN - 1]

    return {
        "reward": head.best_lp,          # the spec's reward: the last head's best value
        "logp": torch.stack(logps, dim=1).sum(dim=1),
        "best_lp": best_lp,              # best seen anywhere, which is what we submit
        "best_pt": best_pt,
        "evals": cfg.evals_per_chain(),
    }


def random_control(sim, h, n_evals, cfg, gen, centre, jitter):
    """Matched-budget control: the same forward passes spent on jittered restarts.

    Without this a chain's score says nothing — the compute inside the surroundings would
    produce a number on its own with no policy at all, and that number is what a learned
    policy has to beat.

    **The control has to be the strong baseline, not a convenient one.** An earlier version
    drew uniformly from a wide box and reported numbers the chain beat; both were sitting
    in a region the schedule probe had already measured as bad, so the comparison flattered
    the policy while both were losing to a single annealing schedule. Starts are now drawn
    around `centre` — the TQA schedule at the measured scale — with the measured `jitter`,
    which is the thing that actually works on this landscape.
    """
    per_start = 1 + 2 * cfg.adam_steps
    starts = max(1, n_evals // per_start)
    n, dev = h.shape[0], h.device
    best_lp = torch.full((n,), -1e30, device=dev)
    best_pt = torch.zeros(n, N_ANGLES, device=dev)
    block = max(1, cfg.chunk // max(n, 1))
    done = 0
    while done < starts:
        b = min(block, starts - done)
        u = torch.randn(n, b, N_ANGLES, device=dev, generator=gen)
        ang = (centre.unsqueeze(1) + u * jitter).reshape(n * b, N_ANGLES)
        hrep = h.repeat_interleave(b, dim=0)
        _, p = refine(sim, hrep, ang, cfg.adam_steps, cfg.lr_gamma, cfg.lr_beta, cfg.chunk)
        lp = p.clamp_min(P_FLOOR).log().view(n, b)
        idx = lp.argmax(dim=1)
        rows = torch.arange(n, device=dev)
        cand_lp, cand_pt = lp[rows, idx], ang.view(n, b, N_ANGLES)[rows, idx]
        better = cand_lp > best_lp
        best_lp = torch.where(better, cand_lp, best_lp)
        best_pt = torch.where(better.unsqueeze(-1), cand_pt, best_pt)
        done += b
    return best_lp, best_pt
