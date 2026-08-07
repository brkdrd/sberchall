import torch
import torch.nn as nn

N_ANGLES = 10          # 5 gamma + 5 beta
TOKEN_DIM = 21         # 10 angles + logP + d(logP)/d(angles)
COND_DIM = 12          # the linear-field vector h


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning (DiT-style).

    The conditioning vector produces per-block shift/scale/gate; gates are
    zero-initialised so every block starts as the identity.
    """

    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, attn_mask):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        y = modulate(self.norm1(x), s1, sc1)
        a, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + g1.unsqueeze(1) * a
        y = modulate(self.norm2(x), s2, sc2)
        x = x + g2.unsqueeze(1) * self.mlp(y)
        return x


class AngleTransformer(nn.Module):
    """Decoder-only transformer over an optimisation trajectory.

    Each token describes one visited point in angle space:
    (10 angles, logP(ground), d logP / d angles). The instance vector h enters
    every block through adaLN-Zero. The output at the last position is a delta
    added to the current angles — the model is a learned optimiser step.
    """

    def __init__(self, d=128, n_heads=4, n_layers=4, max_len=16):
        super().__init__()
        self.inp = nn.Linear(TOKEN_DIM, d)
        self.pos = nn.Parameter(torch.zeros(max_len, d))
        self.cond = nn.Sequential(nn.Linear(COND_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.head = nn.Linear(d, N_ANGLES)
        # zero-init modulation and head: the untrained model proposes delta = 0
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, h):
        T = tokens.shape[1]
        c = self.cond(h)
        x = self.inp(tokens) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, c, mask)
        s, sc = self.ada_out(c).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), s, sc)
        return self.head(x[:, -1])
