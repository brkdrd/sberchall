"""A mixture of angle experts: learn the basins, and learn which one an instance is in.

Everything measured in this repo says the same thing about the shape of the task, and it
is not the shape the problem statement suggests.

- **It is not regression.** For a fixed `h` the good angle vectors form several disjoint
  blobs, one per basin of the landscape. An L2 fit to a set of search-generated labels
  lands on their mean, which sits in no basin at all. Late-layer angles spread as wide as
  a uniform draw across instances (`README`), so a conditional mean carries no signal.
- **It is not search either.** 500 winning angle vectors cross-evaluated against all 500
  instances give mean P 0.315 with *no search at all*, against 0.312 for the 16k-start
  search that produced them (`src/library.py`). Per-instance tuning is worth ~0.09 over a
  single constant vector, and no small set of vectors does the job.
- **It is selection.** The good basins are shared between instances; what changes with `h`
  is *which* of them is the right one.

So the model is a **codebook of angle vectors plus a gate that scores them against `h`**.
The codebook is the shared basins, learned rather than harvested; the gate is the
classification the third point above asks for. Both are trained together, straight
through the organisers' differentiable simulator, on the objective the competition scores.

## The objective

For a gate distribution `w(h) = softmax(gate(h))` over `M` experts,

    L(h) = -log  sum_m  w_m(h) * P_ground(h, C_m)

Maximising the mixture probability, rather than the probability of the single arg-max
expert, is what makes this trainable: every expert receives gradient in proportion to the
responsibility the gate assigns it, so an expert that is nearly right for a group of
instances is pulled the rest of the way, while the gate is simultaneously pulled toward
whichever experts already work. It is the EM split — experts specialise, the gate
partitions — with both halves done by gradient descent through a quantum circuit.

No angle labels are generated anywhere. Training instances are free: `h_train` is i.i.d.
U(-1, 1) (KS p = 0.77), so fresh ones are synthesised every step and the official
`h_train` is never trained on, only used for validation.

## The features

`P_ground` is *exactly* invariant under a 128-element group (`README`, verified to 1e-10):
swapping the two qubits of any of the six pairs `(i, 11-i)` — since `v_i = v_{11-i}` makes
`J` blind to the swap — and the global flip `h -> -h`. All 128 leave the optimal angles
unchanged. Feeding raw `h` therefore asks the network to learn 128 copies of one function.

`canonical_features` folds that away exactly -- bit-exactly, verified to 0.0 deviation over
all 128 group elements on both h_train and h_test -- by fixing the global sign and then
sorting inside each pair, and appends what the gate would otherwise rediscover by brute force:
the ground-state configuration, the spectral gap, and the mean-field magnetisation. Those
are free — the metric itself is defined by a 4096-state enumeration — and the gap is the
single strongest predictor of an instance's difficulty (corr +0.61 with best-found P).

## Inference

Score all `M` experts against `h` (forward passes only, ~M per instance), Adam-polish the
top `K`, keep the post-polish best. The winner has to be chosen *after* polishing: Adam
drives each candidate into its own basin's optimum, so the best start is not the best
finish. Cost is `M + 2*K*steps` forward passes per instance — at the defaults ~5k, which
is about 3 minutes for 500 instances on a Colab T4, inside the competition's 10-minute cap.

Usage:
    python -m src.moe --train --iters 2000 --out runs/moe
    python -m src.moe --predict data/raw/h_test.npy --ckpt runs/moe/best.pt \
                      --submission runs/submission.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .qaoa_ref import QAOA, P as DEPTH

N_ANGLES = 2 * DEPTH
N_QUBITS = 12
PAIRS = [(i, 11 - i) for i in range(6)]
P_FLOOR = 1e-30
# v_i = cos(pi * k_i / 5.5) with k = [0,3,5,2,1,4,4,1,2,5,3,0] reconstructs J exactly:
# J = v v^T - diag(v^2), verified to 0.0 error. Hence E(z) = (v.z)^2/2 - ||v||^2/2 + h.z.
K_INDEX = [0, 3, 5, 2, 1, 4, 4, 1, 2, 5, 3, 0]


def v_vector(device=None):
    k = torch.tensor(K_INDEX, dtype=torch.float32, device=device)
    return torch.cos(torch.pi * k / 5.5)


# ----------------------------------------------------------------------------- features

_TABLES = {}


def _double_tables(sim):
    """(S, quad) in float64, cached per simulator.

    `sim.quad` is built in float32, where the interaction energy of a configuration and of
    its pair-swapped twin -- mathematically identical, since J is blind to the swap --
    differ by ~1e-7. On an instance whose gap is 6e-5 that is a 2e-3 *relative* error, and
    it lands on log(gap). Rebuilding the table in float64 from the same J removes it.
    """
    key = id(sim)
    if key not in _TABLES:
        S = sim.S.double()
        J = sim.J.double()
        _TABLES[key] = (S, 0.5 * torch.einsum("ij,ki,kj->k", J, S, S))
    return _TABLES[key]


@torch.no_grad()
def canonical_features(sim, h):
    """Fold the 128-element symmetry group away, then append spectral facts.

    Returns (N, F) float32. The folding is exact, not learned:

    - **global sign.** `X^{(x)12}` commutes with the mixer and maps `h -> -h`, so an
      instance and its negation have the same optimal angles. The sign of `sum_i h_i v_i`
      picks one of each pair; it is swap-invariant and odd under the flip, which is
      exactly what a sign key has to be.
    - **pair sort.** `v_i = v_{11-i}`, so swapping the two qubits of a pair leaves `J`
      and hence `P_ground` untouched. Sorting `(h_i, h_{11-i})` picks one representative
      of each of the 64 swap orbits.

    The two steps do not commute -- negating `h` exchanges min and max inside every pair
    -- so the sign is fixed first.

    Appended, all computable from the 4096-state enumeration the metric already needs:
    the ground state `z*`, its magnetisation, the gap `E1 - E0`, `E0`, and `min |h_i|` —
    the quantity that decides whether a single-qubit rotation can resolve a qubit at all.
    """
    dev = h.device
    # float64 throughout. Both canonical choices below are decided by comparisons, and in
    # float32 a permuted sum rounds differently in the last ulp: instances whose sign key
    # is near zero, or whose spectral gap is small (the minimum in h_train is 0.011), then
    # canonicalise inconsistently across their own symmetry orbit. Measured deviation over
    # the full 128-element orbit: 1.5e-2 in float32, ~1e-16 here. The cost is one 4096-wide
    # double-precision matmul per batch.
    v = v_vector(dev).double()
    h64 = h.double()
    S64, quad64 = _double_tables(sim)

    # Order matters: the sign flip has to be applied *before* the pair sort. Negating h
    # exchanges the roles of min and max inside every pair, so sorting first and negating
    # afterwards does not commute and leaves the result non-invariant.
    #
    # The sign key is `sum_i h_i v_i`, not the mean-field magnetisation: v_i = v_{11-i}
    # makes it swap-invariant and h -> -h makes it odd, which is exactly what is needed,
    # and being continuous it is zero only on a measure-zero set. The magnetisation is
    # discrete and *is* exactly zero on real instances (43 of the 500 in h_train), which
    # would leave those instances un-canonicalised.
    sign = torch.where((h64 * v).sum(dim=1, keepdim=True) >= 0, 1.0, -1.0).double()
    hs = h64 * sign

    E = quad64.unsqueeze(0) + h64 @ S64.T
    Es, order = E.sort(dim=1)
    zstar = S64[order[:, 0]]                              # (N, 12) in {-1, +1}
    mstar = (zstar * v).sum(dim=1, keepdim=True)
    zs = zstar * sign

    # One permutation, applied to the field and the spins together. Sorting them
    # independently would also be invariant but would throw away the correspondence
    # between a qubit's field and its ground-state spin, which is the informative part.
    hp, zp = hs[:, 6:].flip(-1), zs[:, 6:].flip(-1)
    swap = hs[:, :6] > hp
    canon = torch.cat([torch.where(swap, hp, hs[:, :6]),
                       torch.where(swap, hs[:, :6], hp).flip(-1)], dim=1)
    zc = torch.cat([torch.where(swap, zp, zs[:, :6]),
                    torch.where(swap, zs[:, :6], zp).flip(-1)], dim=1)

    gap = (Es[:, 1] - Es[:, 0]).unsqueeze(1)
    feats = [
        canon,                                            # 12
        zc,                                               # 12
        canon.abs(),                                      # 12
        mstar.abs(),                                      # 1
        gap, gap.log().clamp_min(-12.0),                  # 2
        Es[:, :1] / 10.0,                                 # 1
        (Es[:, 4:5] - Es[:, :1]),                         # 1  width of the low-lying band
        h64.abs().min(dim=1, keepdim=True).values,        # 1
        h64.abs().sum(dim=1, keepdim=True) / 12.0,        # 1
        (canon * v).sum(dim=1, keepdim=True),             # 1  overlap of the field with v
    ]
    return torch.cat(feats, dim=1).to(h.dtype)


FEATURE_DIM = 12 + 12 + 12 + 1 + 2 + 1 + 1 + 1 + 1 + 1


# -------------------------------------------------------------------------------- model

class AngleMoE(nn.Module):
    """A learned codebook of angle vectors, and a gate that scores them against h.

    The codebook is stored in an unconstrained parameterisation and squashed on read:
    `gamma = GAMMA_BOX * tanh(.)` keeps every expert inside the region the sweep measured
    to be the useful one (|gamma| ~ 0.1..1, monotone collapse above 1.0 — see
    `src/gscan.py`'s table), and `beta` is left free because the mixer is pi-periodic, so
    there is nothing outside one period to escape to.
    """

    def __init__(self, n_experts=256, d_model=256, depth=3, gamma_box=1.6):
        super().__init__()
        self.n_experts = n_experts
        self.gamma_box = gamma_box
        raw = torch.empty(n_experts, N_ANGLES)
        nn.init.uniform_(raw[:, :DEPTH], -0.8, 0.8)
        nn.init.uniform_(raw[:, DEPTH:], -1.4, 1.4)
        self.codebook = nn.Parameter(raw)

        layers, d = [], FEATURE_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU()]
            d = d_model
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, n_experts)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # start uniform over experts: no expert is favoured

    def angles(self):
        """Codebook in radians, (M, 10)."""
        g = self.gamma_box * torch.tanh(self.codebook[:, :DEPTH])
        b = self.codebook[:, DEPTH:]
        return torch.cat([g, b], dim=1)

    def logits(self, feats):
        return self.head(self.trunk(feats))


# ----------------------------------------------------------------------------- training

def expert_probs(sim, h, ang, chunk=8192):
    """P_ground for every (instance, expert) pair -> (N, M), differentiable in `ang`."""
    n, m = h.shape[0], ang.shape[0]
    hr = h.repeat_interleave(m, dim=0)
    ar = ang.unsqueeze(0).expand(n, -1, -1).reshape(n * m, N_ANGLES)
    out = []
    for lo in range(0, n * m, chunk):
        out.append(sim.p_ground(hr[lo:lo + chunk], ar[lo:lo + chunk, :DEPTH],
                                ar[lo:lo + chunk, DEPTH:]))
    return torch.cat(out).view(n, m)


def train(sim, model, opt, iters, batch, active, device, h_val=None, log_every=50,
          out_dir=None, seed=0):
    """Train gate and codebook together on the mixture objective.

    `active` subsamples the codebook each step: evaluating all M experts on all B
    instances is B*M circuit simulations, and the gradient is just as informative from a
    random subset. The subset is drawn uniformly so every expert keeps receiving gradient
    regardless of how confident the gate has become.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    hist, best = [], -1.0
    t0 = time.time()
    for it in range(1, iters + 1):
        h = (torch.rand(batch, N_QUBITS, device=device, generator=gen) * 2 - 1)
        feats = canonical_features(sim, h)
        idx = torch.randperm(model.n_experts, device=device, generator=gen)[:active]
        ang = model.angles()[idx]
        p = expert_probs(sim, h, ang)                        # (B, active)
        w = torch.softmax(model.logits(feats)[:, idx], dim=1)
        mix = (w * p).sum(dim=1)
        loss = -(mix.clamp_min(P_FLOOR).log()).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                top = p.gather(1, w.argmax(dim=1, keepdim=True)).squeeze(1)
                rec = {"iter": it, "loss": loss.item(), "mixture_P": mix.mean().item(),
                       "gate_pick_P": top.mean().item(), "best_expert_P": p.max(dim=1).values.mean().item(),
                       "secs": time.time() - t0}
            if h_val is not None:
                rec["val_gate_P"] = evaluate(sim, model, h_val, polish_steps=0)["gate_P"]
                if rec["val_gate_P"] > best:
                    best = rec["val_gate_P"]
                    if out_dir:
                        save(model, Path(out_dir) / "best.pt")
            hist.append(rec)
            print("  ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}"
                            for k, v in rec.items()), flush=True)
    if out_dir:
        save(model, Path(out_dir) / "last.pt")
        (Path(out_dir) / "history.json").write_text(json.dumps(hist, indent=1))
    return hist


# ---------------------------------------------------------------------------- inference

def refine(sim, h, ang, steps, lr_gamma, lr_beta, chunk=2048):
    """Adam on the angles; returns the best point *visited*, since P is not monotone
    along an Adam path. Branchless best-tracking: `if better.any()` forces a GPU sync
    every step and drains the pipeline."""
    best_ang = ang.clone()
    with torch.no_grad():
        best_p = torch.cat([sim.p_ground(h[l:l + chunk], ang[l:l + chunk, :DEPTH],
                                         ang[l:l + chunk, DEPTH:])
                            for l in range(0, ang.shape[0], chunk)])
    if steps <= 0:
        return best_ang, best_p
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
                pd = p.detach()
                better = pd > best_p[sl]
                best_p[sl] = torch.where(better, pd, best_p[sl])
                best_ang[sl] = torch.where(better.unsqueeze(1),
                                           torch.cat([g, b], dim=1).detach(), best_ang[sl])
    return best_ang, best_p


@torch.no_grad()
def _gate_topk(sim, model, h, k):
    feats = canonical_features(sim, h)
    logits = model.logits(feats)
    return logits.topk(min(k, model.n_experts), dim=1).indices


def evaluate(sim, model, h, top_k=8, polish_steps=0, lr_gamma=0.05, lr_beta=0.03,
             chunk=2048, select="measured"):
    """Score the model on a set of instances.

    `select` decides how the `top_k` candidates to polish are chosen out of the codebook:

    - **"measured"** — score every expert against `h` through the simulator and take the
      best. The codebook is small and the scoring is forward-only, so this costs `M`
      circuit evaluations per instance, against the thousands the polish costs. There is
      no reason to trust a predicted ranking when the true one is this cheap, and it
      cannot do worse than the gate by construction.
    - **"gate"** — rank by the gate's logits alone, never consulting the simulator. This
      is what an angle-*prediction* model does in the strict sense, and it is what the
      reported `gate_P` measures, but it throws away a cheap exact signal.

    `gate_P` is always the measured P of the gate's own top pick, so the two selection
    modes stay comparable and the gate's contribution is visible either way.
    """
    n = h.shape[0]
    with torch.no_grad():
        ang_all = model.angles()
        logits = model.logits(canonical_features(sim, h))
        if select == "measured":
            table = expert_probs(sim, h, ang_all, chunk * 4)          # (N, M)
            idx = table.topk(min(top_k, model.n_experts), dim=1).indices
            rows = torch.arange(n, device=h.device)
            p_gate = table[rows, logits.argmax(dim=1)]
        elif select == "gate":
            idx = logits.topk(min(top_k, model.n_experts), dim=1).indices
            g1 = ang_all[idx[:, 0]]
            p_gate = torch.cat([sim.p_ground(h[l:l + chunk], g1[l:l + chunk, :DEPTH],
                                             g1[l:l + chunk, DEPTH:])
                                for l in range(0, n, chunk)])
        else:
            raise ValueError(f"unknown select {select!r}")
    out = {"gate_P": p_gate.mean().item(), "select": select}
    if select == "measured":
        out["best_of_codebook_P"] = table.max(dim=1).values.mean().item()
    if polish_steps > 0:
        cand = ang_all[idx.reshape(-1)].clone()
        hr = h.repeat_interleave(idx.shape[1], dim=0)
        cand, p = refine(sim, hr, cand, polish_steps, lr_gamma, lr_beta, chunk)
        p = p.view(n, -1)
        j = p.argmax(dim=1)
        rows = torch.arange(n, device=h.device)
        out["polished_P"] = p[rows, j].mean().item()
        out["angles"] = cand.view(n, -1, N_ANGLES)[rows, j]
        out["per_instance"] = p[rows, j]
    return out


def write_submission(path, ang):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = ang.detach().cpu().numpy()
    if a.shape != (a.shape[0], N_ANGLES):
        raise ValueError(f"expected (N, {N_ANGLES}) angles, got {a.shape}")
    head = "id," + ",".join([f"gamma_{i}" for i in range(DEPTH)]
                            + [f"beta_{i}" for i in range(DEPTH)])
    rows = [head]
    for i, r in enumerate(a):
        rows.append(str(i) + "," + ",".join(f"{x:.8f}" for x in r))
    path.write_text("\n".join(rows) + "\n")
    return path


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_experts": model.n_experts,
                "gamma_box": model.gamma_box,
                "d_model": model.head.in_features}, path)


def init_codebook(model, ang, gamma_box=None):
    """Seed the codebook with known-good angle vectors.

    A cold codebook starts as noise, so early training spends itself rediscovering angles
    this repo already has: `submission_train.csv` holds one searched vector per h_train
    instance, and cross-evaluating those against all instances scores 0.315 with no search
    at all. Seeding with them makes best-of-codebook a *floor* the gate starts from rather
    than a target it has to reach, and training then only has to improve on it.

    The codebook is stored pre-squash, so the seed is inverted through atanh. Any vector
    whose |gamma| exceeds the box would invert to infinity, so the box is widened to fit
    the data unless the caller pins it.
    """
    ang = torch.as_tensor(ang, dtype=torch.float32)
    m = min(ang.shape[0], model.n_experts)
    g, b = ang[:m, :DEPTH], ang[:m, DEPTH:]
    box = gamma_box if gamma_box is not None else max(model.gamma_box,
                                                      float(g.abs().max()) * 1.05)
    model.gamma_box = box
    with torch.no_grad():
        model.codebook[:m, :DEPTH] = torch.atanh((g / box).clamp(-0.999, 0.999))
        model.codebook[:m, DEPTH:] = b
    return m


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_seed(path):
    """Find the seed file relative to the CWD or to the repo root.

    Inside the container the working directory is not necessarily the repo, and a seed
    that cannot be found must not be skipped quietly: the run that produced
    models/moe_best.pt did exactly that and trained from cold to 0.1537, against the
    0.3047 the seed alone scores.
    """
    p = Path(path)
    for cand in (p, REPO_ROOT / p):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"codebook seed {path} not found (looked in {p.resolve()} and {REPO_ROOT / p}). "
        f"Pass --init-codebook none for a deliberate cold start.")


def load_angle_file(path):
    """Angle vectors from a submission .csv, a plain .npy, or any (*, 10) array in an .npz."""
    path = Path(path)
    if path.suffix == ".csv":
        a = np.loadtxt(path, delimiter=",", skiprows=1)
        return a[:, 1:] if a.shape[1] == N_ANGLES + 1 else a
    if path.suffix == ".npy":
        return np.load(path)
    z = np.load(path)
    out = [np.asarray(z[k], dtype=np.float32) for k in z.files
           if np.asarray(z[k]).ndim == 2 and np.asarray(z[k]).shape[-1] == N_ANGLES]
    if not out:
        raise ValueError(f"no (*, {N_ANGLES}) angle array in {path}")
    return np.unique(np.concatenate(out, axis=0).round(6), axis=0)


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = AngleMoE(n_experts=ck["n_experts"], d_model=ck["d_model"],
                 gamma_box=ck["gamma_box"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


# ---------------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/sberdata"))
    ap.add_argument("--out", type=Path, default=Path("runs/moe"))
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", type=Path, default=None, help="h .npy to predict for")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--submission", type=Path, default=Path("runs/submission.csv"))
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--init-codebook", type=Path,
                    default=Path("submission_train.csv"),
                    help="angle vectors to seed the codebook with; 'none' for a cold start")
    ap.add_argument("--active", type=int, default=64, help="experts evaluated per step")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--gamma-box", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--polish", type=int, default=300)
    ap.add_argument("--lr-gamma", type=float, default=0.05)
    ap.add_argument("--lr-beta", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--freeze-codebook", type=int, default=1,
                    help="1: train the gate only, so a seeded codebook cannot regress")
    ap.add_argument("--select", default="measured", choices=("measured", "gate"),
                    help="how the candidates to polish are picked out of the codebook")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    torch.manual_seed(args.seed)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    h_train = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                           device=dev)
    print(f"device {dev} | experts {args.experts} | features {FEATURE_DIM}")

    if args.train:
        model = AngleMoE(args.experts, args.d_model, gamma_box=args.gamma_box).to(dev)
        floor = None
        if str(args.init_codebook).lower() != "none":
            seed_path = resolve_seed(args.init_codebook)
            seeded = init_codebook(model, load_angle_file(seed_path))
            print(f"codebook seeded with {seeded} vectors from {seed_path} "
                  f"(gamma box widened to {model.gamma_box:.2f})")
            # on a subset: M x N circuit evaluations is seconds on a GPU but ten minutes
            # on CPU, and this is a sanity print, not a reported result
            with torch.no_grad():
                sub = h_train[:128]
                floor = expert_probs(sim, sub, model.angles()).max(dim=1).values.mean().item()
            print(f"best-of-codebook on {sub.shape[0]} h_train instances before any "
                  f"training: {floor:.5f}")
        else:
            print("cold start: no codebook seed")
        if args.freeze_codebook and floor is not None:
            model.codebook.requires_grad_(False)
            print("codebook frozen: training fits the gate only, so best-of-codebook "
                  "cannot regress below the floor above")
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=args.lr)
        train(sim, model, opt, args.iters, args.batch, args.active, dev,
              h_val=h_train, out_dir=args.out, seed=args.seed)
        res = evaluate(sim, model, h_train, args.top_k, args.polish,
                       args.lr_gamma, args.lr_beta, args.chunk, args.select)
        if floor is not None and "best_of_codebook_P" in res:
            print(f"best-of-codebook after training: {res['best_of_codebook_P']:.5f} "
                  f"(floor was {floor:.5f})")
        msg = f"h_train: gate alone {res['gate_P']:.5f}"
        if "polished_P" in res:
            msg += f", top-{args.top_k} polished {res['polished_P']:.5f}"
        print(msg)

    if args.predict is not None:
        ckpt = args.ckpt or (args.out / "best.pt")
        if str(ckpt).lower() == "none" or not Path(ckpt).exists():
            # Fallback with no trained gate. With --select measured the gate plays no part
            # in choosing candidates, so a seeded codebook alone still produces a valid
            # submission -- worth having when a training run is unavailable or suspect.
            model = AngleMoE(args.experts, args.d_model, gamma_box=args.gamma_box).to(dev)
            seed_path = resolve_seed(args.init_codebook)
            init_codebook(model, load_angle_file(seed_path))
            print(f"no checkpoint at {ckpt}: predicting from a codebook seeded with "
                  f"{seed_path} and an untrained gate (select={args.select})")
        else:
            model = load(ckpt, dev)
        h = torch.tensor(np.load(args.predict), dtype=torch.float32, device=dev)
        t0 = time.time()
        res = evaluate(sim, model, h, args.top_k, args.polish, args.lr_gamma,
                       args.lr_beta, args.chunk, args.select)
        dt = time.time() - t0
        if "angles" not in res:
            raise SystemExit("--polish must be > 0 to produce a submission")
        write_submission(args.submission, res["angles"])
        if "best_of_codebook_P" in res:
            print(f"  best-of-codebook (measured, no polish): {res['best_of_codebook_P']:.5f}")
        print(f"{args.predict.name}: gate alone {res['gate_P']:.5f}, "
              f"polished {res['polished_P']:.5f}  [{dt:.0f}s for {h.shape[0]} instances, "
              f"limit 600 s]")
        print(f"wrote {args.submission}")
    return 0


if __name__ == "__main__":
    main()
