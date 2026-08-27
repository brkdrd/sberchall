"""Angle parameterisation: the model works in normalised units, the simulator in radians.

The two halves of the angle vector live on wildly different scales, and treating them as
one homogeneous 10-vector is a bug that shows up everywhere at once.

**gamma.** The phase separator applies `exp(i * gamma * E)`. For this `J` with
`h ~ U(-1, 1)` the spectrum spans about 39, so the phase completes a full revolution by
`gamma ~ 2*pi/39 ~ 0.16`. Beyond that the phase wraps and the landscape is chaotic rather
than merely non-convex.

**beta.** The mixer `exp(-i * beta * sum_j X_j)` is periodic in `beta` with period `pi`,
independent of the problem, so the whole useful range is `(-pi/2, pi/2)`.

Sampling both uniformly on `(-pi/2, pi/2)` — as this pipeline did — means gamma is drawn
from a box roughly 10x wider than the region that carries signal. In five gamma dimensions
the useful fraction of a `(-pi, pi)^5` box is `(0.16/pi)^5 ~ 3e-7`: a 65k-point screen
expects 0.02 hits. It also mis-scales two things that are easy to miss:

- a single optimiser step size is simultaneously far too large for gamma and too small for
  beta (`lr=0.03` is 2% of beta's range but 19% of gamma's);
- `d logP/d gamma` carries a factor of `E ~ +-20` relative to `d logP/d beta`, so a shared
  gradient clip saturates the gamma features and feeds the model a dead input.

Working in normalised units `u`, with `angles = u * ANGLE_SCALE`, fixes all three: a unit
step means the same thing in both halves, and `d logP/du = scale * d logP/d angle` puts the
gradient features on a common scale by construction.
"""

import torch

from .qaoa_ref import P as DEPTH

N_ANGLES = 2 * DEPTH


def energy_span(sim, h, chunk=256):
    """Mean per-instance spread of the cost Hamiltonian spectrum, max(E) - min(E)."""
    spans = []
    for lo in range(0, h.shape[0], chunk):
        E = sim.energies(h[lo:lo + chunk])
        spans.append(E.max(dim=1).values - E.min(dim=1).values)
    return torch.cat(spans).mean().item()


def angle_scale(sim, h, device=None):
    """Per-coordinate scale making a unit step meaningful in both halves.

    gamma -> 2*pi / span(E): one full revolution of the phase factor.
    beta  -> pi/2:           half of the mixer's period, so u in (-1, 1) is one period.

    Returns a (10,) tensor to multiply normalised units by.
    """
    span = energy_span(sim, h)
    g = 2.0 * torch.pi / span
    b = torch.pi / 2.0
    dev = device if device is not None else h.device
    return torch.tensor([g] * DEPTH + [b] * DEPTH, dtype=torch.float32, device=dev)


def legacy_scale(device=None):
    """Identity scale: reproduces the old behaviour for checkpoints trained in radians."""
    return torch.ones(N_ANGLES, dtype=torch.float32, device=device)


def canonicalise(u):
    """Fold normalised angles into the fundamental domain of P(ground)'s symmetry group.

    P(ground) is *exactly* invariant under a 64-element group acting on the angles:

    - `beta_i += pi` independently per layer (the mixer flips every qubit's sign, and
      `(-1)^12 = +1`), which in normalised units is `u_beta += 2`  -> 2^5 = 32 elements;
    - the global sign flip `(gamma, beta) -> (-gamma, -beta)`, which conjugates a state
      built from a real spectrum and so leaves every amplitude's modulus alone -> x2.

    So every optimum comes with 63 exact duplicates. Averaging over them — which is what
    an MSE regression to a set of search-generated labels does — lands on their mean,
    and the mean over a `+-(gamma, beta)` pair is zero. Folding the labels into one
    representative first is what makes the target a function of `h` again.

    Canonical form: flip the global sign so that the largest-magnitude gamma is positive,
    then fold every beta into one period.
    """
    u = u.clone()
    g, b = u[..., :DEPTH], u[..., DEPTH:]
    lead = g.gather(-1, g.abs().argmax(dim=-1, keepdim=True))
    u = torch.where(lead < 0, -u, u)
    g, b = u[..., :DEPTH], u[..., DEPTH:]
    return torch.cat([g, (b + 1.0) % 2.0 - 1.0], dim=-1)


def to_angles(u, scale):
    """Normalised units -> radians for the simulator."""
    return u * scale


def to_units(angles, scale):
    """Radians -> normalised units."""
    return angles / scale
