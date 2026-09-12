"""The two networks: an MLP that places the root node, a transformer that moves the head.

Both are policies, not regressors. Nothing here is trained against an angle label — the
only signal is the value the chain finishes on, delivered by REINFORCE (`src/reinforce.py`).

**RootMLP** maps `h -> a point in angle space`. It is the only part of the system that
sees `h` and nothing else, so it carries whatever is learnable about an instance before
any evaluation has been made.

**NodePolicy** is a decoder-only transformer over the *surroundings* of at most five
nodes — the current head, its parent, its grandparent, and whichever children of the head
already exist — and emits the offset to the next child. One token per sampled point, so a
node with `k` samples contributes `k` tokens and the model sees individual probes rather
than a summary of them.

Three choices worth stating, because each is load-bearing:

- **Positions in tokens are relative to the head.** The offset the model emits is relative
  too, so the whole prediction is equivariant to where the head sits: a configuration of
  probes that means "go north-east" means the same thing at any point in angle space. The
  absolute position enters once, through the conditioning vector, because the landscape is
  *not* translation invariant (gamma = 0 is a special point) and the policy has to be able
  to know where it is.
- **Roles are an embedding, not a position.** There is no sequence here — the context is a
  set of probes, each tagged by which node it belongs to. Ordering them would invent
  structure that is not in the data. A learned readout token replaces "take the last
  position".
- **The offset head is zero-initialised** (adaLN-Zero convention, as in `model.py`). An
  untrained policy therefore proposes offset 0, so the chain begins as a random walk of
  scale `sigma` rather than as a random jump of unbounded size: the run degrades to
  structured multistart instead of to noise.
"""

import torch
import torch.nn as nn

from .model import Block, modulate
from .qaoa_ref import P as DEPTH

N_ANGLES = 2 * DEPTH
H_DIM = 12

# roles a context node can have
HEAD, PARENT, GRAND, CHILD1, CHILD2 = range(5)
N_ROLES = 5

# one token per probe: four position groups (start, finish, displacement, node centre),
# all relative to the head, plus the two values and the improvement between them
TOKEN_DIM = 4 * N_ANGLES + 3
# h, head centre (absolute), head's best value, depth through the chain, which child
COND_DIM = H_DIM + N_ANGLES + 3


def mlp(d_in, d_hidden, d_out, layers=2):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, d_hidden), nn.SiLU()]
        d = d_hidden
    mods += [nn.Linear(d, d_out)]
    return nn.Sequential(*mods)


class RootMLP(nn.Module):
    """h -> the root node's centre, in radians.

    The final layer is zero-initialised and the bias holds a smooth annealing schedule, so
    an untrained root sits on the TQA family (Sack & Serbyn) rather than at a random
    corner of angle space. Training moves it from there; `gamma_top` and `beta_top` say
    where "there" is and are hyperparameters, not a constraint — the layer above is free
    to learn any offset from them.
    """

    def __init__(self, d=256, layers=3, gamma_top=1.0, beta_top=0.8):
        super().__init__()
        self.net = mlp(H_DIM, d, N_ANGLES, layers)
        nn.init.zeros_(self.net[-1].weight)
        l = torch.arange(1, DEPTH + 1, dtype=torch.float32)
        schedule = torch.cat([(l / DEPTH) * gamma_top, (1.0 - l / DEPTH) * beta_top])
        with torch.no_grad():
            self.net[-1].bias.copy_(schedule)

    def forward(self, h):
        return self.net(h)


class NodePolicy(nn.Module):
    """Surroundings of up to five nodes -> the offset from the head to its next child."""

    def __init__(self, d=128, n_heads=4, n_layers=4):
        super().__init__()
        self.inp = nn.Linear(TOKEN_DIM, d)
        self.role = nn.Embedding(N_ROLES, d)
        self.readout = nn.Parameter(torch.zeros(1, 1, d))
        self.cond = nn.Sequential(nn.Linear(COND_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.head = nn.Linear(d, N_ANGLES)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, roles, valid, cond):
        """tokens (B,T,TOKEN_DIM), roles (B,T) long, valid (B,T) bool, cond (B,COND_DIM)."""
        c = self.cond(cond)
        x = self.inp(tokens) + self.role(roles)
        x = torch.cat([self.readout.expand(x.shape[0], -1, -1), x], dim=1)
        # the readout token is always visible; padded probe slots never are
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)
        for blk in self.blocks:
            x = blk(x, c, None, pad)
        s, sc = self.ada_out(c).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), s, sc)
        return self.head(x[:, 0])
