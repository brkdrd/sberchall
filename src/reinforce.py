"""REINFORCE over the node chain: train the two policies on the value a chain finishes on.

There are no angle labels in this competition and none are invented here. A chain is a
sequence of actions — the root MLP's point, then three offsets per head — and the whole
sequence is scored once, by the best P(ground) in the final head's surroundings. That
single number trains everything:

    loss = -sum_t log pi(a_t | context_t) * (R - baseline)

which is the score-function estimator, and it is the right one rather than the convenient
one. The alternative — backpropagating pathwise through the chain, which the
reparameterisation `node = parent + offset` does make possible — dies on the inner Adam
loop: `start -> finish` is piecewise constant in value, so its derivative is zero inside a
basin and undefined at the boundary between two. The event the policy has to learn is
"which basin does this node fall into", and that event is exactly the one a pathwise
gradient cannot see.

**Baseline.** Not a change to the estimator, just variance reduction: `chains` chains are
run per instance and each is scored against the mean of its siblings (leave-one-out, so
the baseline is independent of the chain it corrects and the estimator stays unbiased).
With a single terminal reward spread over ~13 actions there is no getting away without
one.

**The control is part of the result.** Every evaluation also runs `random_control` at a
matched forward-pass budget. Most of a chain's compute is in the Adam refinement inside
its surroundings, and that produces good angles whether or not a policy chose where to
put them. The number that means anything is the gap between the two.

Usage:
    python -m src.reinforce --train-iters 2000 --batch 32 --chains 8
    python -m src.reinforce --predict data/raw/h_test.npy --ckpt runs/best.pt
"""

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from .chain import ChainConfig, N_ANGLES, random_control, rollout
from .qaoa_ref import P as DEPTH
from .refine import refine
from .policy import NodePolicy, RootMLP
from .predict import write_submission
from .qaoa_ref import QAOA

H_DIM = 12


def synth(batch, device, gen):
    """Fresh instances. h_train is i.i.d. U(-1, 1), so training data is free and the
    official 500 stay held out."""
    return torch.rand(batch, H_DIM, device=device, generator=gen) * 2 - 1


ROOT_LADDER = (0.16, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def probe_root_scale(sim, h, cfg, beta_top, ladder=ROOT_LADDER, steps=40):
    """Measure where the root should start instead of assuming it.

    The root MLP's bias is a TQA schedule, and `gamma_top` sets its scale. That scale is
    the open question: `angles.py` puts it at 2*pi/span(E) ~ 0.16 rad, while the gap the
    metric actually has to resolve (E1-E0 ~ 0.17) asks for gamma ~ pi/0.17 ~ 18. Rather
    than pick a side, score the bare schedule plus a short Adam polish at each scale and
    start the policy at whichever wins. Costs one pass over a handful of instances.
    """
    dev = h.device
    l = torch.arange(1, DEPTH + 1, device=dev, dtype=torch.float32)
    rows = []
    for g in ladder:
        ang = torch.cat([(l / DEPTH) * g, (1.0 - l / DEPTH) * beta_top])
        ang = ang.unsqueeze(0).expand(h.shape[0], -1).contiguous()
        _, p = refine(sim, h, ang, steps, cfg.lr_gamma, cfg.lr_beta, cfg.chunk)
        rows.append((g, p.mean().item()))
    best = max(rows, key=lambda r: r[1])
    print("root scale probe (TQA schedule + " + str(steps) + " Adam steps, "
          + str(h.shape[0]) + " instances):")
    for g, m in rows:
        print(f"    gamma_top {g:6.2f}  mean P {m:.5f}" + ("   <-" if g == best[0] else ""))
    return best


def build(args, device):
    root = RootMLP(d=args.root_dim, layers=args.root_layers,
                   gamma_top=args.root_gamma_top, beta_top=args.root_beta_top).to(device)
    policy = NodePolicy(d=args.d_model, n_heads=args.heads,
                        n_layers=args.layers).to(device)
    return root, policy


def chain_config(args):
    return ChainConfig(**{f.name: getattr(args, f.name) for f in fields(ChainConfig)})


def infer_config(args, cfg):
    """The chain used at inference. Bigger than the training chain, same weights."""
    out = ChainConfig(**asdict(cfg))
    for name in ("k", "iters", "adam_steps"):
        v = getattr(args, f"infer_{name}")
        if v:
            setattr(out, name, v)
    return out


@torch.no_grad()
def evaluate(root, policy, sim, h, cfg, gen, chains, control=True):
    """Best-of-`chains` per instance, plus the matched-budget random control."""
    n, dev = h.shape[0], h.device
    rows = torch.arange(n, device=dev)
    hrep = h.repeat_interleave(chains, dim=0)
    t0 = time.time()
    out = rollout(root, policy, sim, hrep, cfg, gen)
    secs = time.time() - t0
    lp = out["best_lp"].view(n, chains)
    pt = out["best_pt"].view(n, chains, N_ANGLES)
    idx = lp.argmax(dim=1)
    best_lp, best_pt = lp[rows, idx], pt[rows, idx]

    res = {"mean_p": best_lp.exp().mean().item(),
           "median_p": best_lp.exp().median().item(),
           "terminal_p": out["reward"].exp().mean().item(),
           "evals_per_instance": chains * cfg.evals_per_chain(),
           "seconds": secs, "angles": best_pt}
    if control:
        centre = root(h)
        c_lp, _ = random_control(sim, h, res["evals_per_instance"], cfg, gen, centre)
        res["control_p"] = c_lp.exp().mean().item()
    return res


def train(args, root, policy, sim, h_eval, cfg, device, out_dir):
    gen = torch.Generator(device=device).manual_seed(args.seed)
    opt = torch.optim.Adam(list(root.parameters()) + list(policy.parameters()), lr=args.lr)
    best, history, t0 = -1.0, [], time.time()

    print(f"chain: {1 + 3 * cfg.iters} nodes x {cfg.k} probes x "
          f"(1 + 2*{cfg.adam_steps}) = {cfg.evals_per_chain()} forward passes")
    print(f"iteration: {args.batch} instances x {args.chains} chains = "
          f"{args.batch * args.chains * cfg.evals_per_chain() / 1e6:.2f}M forward passes\n")

    for it in range(1, args.train_iters + 1):
        h = synth(args.batch, device, gen)
        hrep = h.repeat_interleave(args.chains, dim=0)
        out = rollout(root, policy, sim, hrep, cfg, gen)

        r = out["reward"].view(args.batch, args.chains).detach()
        if args.chains > 1:
            base = (r.sum(dim=1, keepdim=True) - r) / (args.chains - 1)
        else:
            base = r.mean()
        adv = r - base
        if args.adv_norm:
            adv = adv / adv.std().clamp_min(1e-6)
        loss = -(out["logp"].view(args.batch, args.chains) * adv).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            list(root.parameters()) + list(policy.parameters()), args.clip)
        opt.step()

        if it % args.log_every == 0:
            print(f"  it {it:5d}  R {r.mean().item():8.3f}  "
                  f"P(terminal) {r.exp().mean().item():.5f}  "
                  f"|adv| {adv.abs().mean().item():7.3f}  "
                  f"grad {gnorm.item():8.2f}  [{(time.time() - t0) / 60:.1f} min]")

        if it % args.eval_every == 0 or it == args.train_iters:
            ev = evaluate(root, policy, sim, h_eval, cfg, gen, args.eval_chains)
            ev.pop("angles")
            print(f"  eval@{it}: mean P {ev['mean_p']:.5f} | control "
                  f"{ev.get('control_p', float('nan')):.5f} | "
                  f"{ev['evals_per_instance']} evals/instance | {ev['seconds']:.0f}s")
            history.append({"iter": it, **ev})
            if ev["mean_p"] > best:
                best = ev["mean_p"]
                torch.save({"root": root.state_dict(), "policy": policy.state_dict(),
                            "cfg": asdict(cfg), "args": vars(args), "mean_p": best},
                           out_dir / "best.pt")
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"  wall-clock cap reached at iteration {it}")
            break

    torch.save({"root": root.state_dict(), "policy": policy.state_dict(),
                "cfg": asdict(cfg), "args": vars(args)}, out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_mean_p": best, "history": history,
            "minutes": (time.time() - t0) / 60}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/reinforce"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--predict", type=Path, default=None,
                    help="h*.npy to predict for; skips training and writes a submission")
    ap.add_argument("--submission", type=Path, default=None)
    # training
    ap.add_argument("--train-iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32, help="instances per iteration")
    ap.add_argument("--chains", type=int, default=8,
                    help="chains per instance; also the leave-one-out baseline's sample")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--adv-norm", action="store_true",
                    help="divide the advantage by its batch std (biased, steadier)")
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-instances", type=int, default=128)
    ap.add_argument("--eval-chains", type=int, default=4)
    # models
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--root-dim", type=int, default=256)
    ap.add_argument("--root-layers", type=int, default=3)
    ap.add_argument("--root-gamma-top", type=float, default=1.0)
    ap.add_argument("--probe-root", type=int, default=64,
                    help="instances used to measure --root-gamma-top; 0 trusts the flag")
    ap.add_argument("--root-beta-top", type=float, default=0.8)
    # the chain itself — every ChainConfig field is a flag
    for f in fields(ChainConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default),
                        default=f.default)
    # inference runs a bigger chain than training: the context is addressed by role, not
    # by position, so neither the probe count nor the chain length is baked into the model
    ap.add_argument("--infer-k", type=int, default=0, help="0 = same as training")
    ap.add_argument("--infer-iters", type=int, default=0)
    ap.add_argument("--infer-adam-steps", type=int, default=0)
    ap.add_argument("--infer-chains", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dev = args.device
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim = QAOA(np.load(args.data_dir / "J.npy"), device=dev)
    cfg = chain_config(args)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    if args.probe_root and args.ckpt is None:
        probe_h = synth(args.probe_root, dev, gen)
        g, m = probe_root_scale(sim, probe_h, cfg, args.root_beta_top)
        print(f"  -> root_gamma_top = {g} (was {args.root_gamma_top}), mean P {m:.5f}\n")
        args.root_gamma_top = g

    if args.ckpt is None:
        root, policy = build(args, dev)
    else:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        # the checkpoint carries the shape it was built at, so predicting from it does not
        # mean retyping every architecture flag. The *chain* config deliberately does not
        # come along: inference runs a bigger chain than training on the same weights.
        for k in ("d_model", "layers", "heads", "root_dim", "root_layers",
                  "root_gamma_top", "root_beta_top"):
            if k in ck.get("args", {}):
                setattr(args, k, ck["args"][k])
        root, policy = build(args, dev)
        root.load_state_dict(ck["root"])
        policy.load_state_dict(ck["policy"])
        root.eval()
        policy.eval()
        print(f"loaded {args.ckpt} (mean P {ck.get('mean_p', float('nan')):.5f})")

    if args.predict is not None:
        h = torch.tensor(np.load(args.predict), dtype=torch.float32, device=dev)
        t0 = time.time()
        icfg = infer_config(args, cfg)
        chains = args.infer_chains or args.eval_chains
        ev = evaluate(root, policy, sim, h, icfg, gen, chains, control=False)
        path = args.submission or (args.out_dir / "submission.csv")
        write_submission(path, ev["angles"])
        print(f"mean P(ground) {ev['mean_p']:.5f} on {h.shape[0]} instances in "
              f"{time.time() - t0:.0f}s ({ev['evals_per_instance']} evals/instance); "
              f"the limit is 600 s")
        return {"mean_p": ev["mean_p"], "seconds": time.time() - t0}

    h_train = torch.tensor(np.load(args.data_dir / "h_train.npy"), dtype=torch.float32,
                           device=dev)
    h_eval = h_train[:args.eval_instances]
    res = train(args, root, policy, sim, h_eval, cfg, dev, args.out_dir)

    ck = torch.load(args.out_dir / "best.pt", map_location=dev, weights_only=False)
    root.load_state_dict(ck["root"])
    policy.load_state_dict(ck["policy"])
    icfg = infer_config(args, cfg)
    final = evaluate(root, policy, sim, h_train, icfg, gen,
                     args.infer_chains or args.eval_chains)
    write_submission(args.out_dir / "submission_train.csv", final.pop("angles"))
    print(f"\nfull h_train: mean P {final['mean_p']:.5f} | control "
          f"{final['control_p']:.5f} | {final['seconds']:.0f}s for 500 instances "
          f"(limit 600 s)")
    res["final"] = final
    (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
