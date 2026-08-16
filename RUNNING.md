# Running the baseline

The notebooks are self-contained: no repo dependency — no clone, no `pip install` — so
`kaggle kernels push` only needs the notebook itself. All of them need a GPU.

| notebook | what it is |
|---|---|
| `01_baseline_mlp.ipynb` | supervised MLP on generated angle labels |
| `02_direct_optimization_baseline.ipynb` | Adam on the angles, 32 random restarts — no model |
| `03_massive_multistart.ipynb` | screen ~65k Sobol starts per instance, Adam-refine the survivors |
| `04_cma_es_restarts.ipynb` | batched IPOP-CMA-ES with restarts, then an Adam polish |

Notebooks 03 and 04 are the two strongest searches and the ones to run for a reference score; see
[Massive multistart](#massive-multistart-notebook-03) and
[CMA-ES with restarts](#cma-es-with-restarts-notebook-04) below.

## What it needs

The three organiser files, discoverable at runtime:

| file | purpose |
|---|---|
| `J.npy` | fixed 12×12 Ising coupling matrix |
| `h_train.npy` | 500 unlabelled linear-field vectors |
| `QAOA.py` | reference simulator — defines the metric |

The notebook's `find_file()` searches `/kaggle/input/**`, `/kaggle/working`, `./data/raw`,
`../data/raw`, `.`, `..`, `/content` — so any Kaggle dataset slug works without editing paths.
It picks up `h_test.npy` the same way once that is released, and otherwise falls back to
`h_train.npy` so the submission cell stays runnable.

## Kaggle

Templates are in `kaggle/`; replace `USERNAME` with your Kaggle handle in both files.

```bash
pip install kaggle                      # credentials at ~/.kaggle/kaggle.json, chmod 600

# 1. upload the three organiser files once as a private dataset
mkdir -p /tmp/qaoa-data && cp data/raw/J.npy data/raw/h_train.npy src/qaoa_ref.py /tmp/qaoa-data/
mv /tmp/qaoa-data/qaoa_ref.py /tmp/qaoa-data/QAOA.py
cp kaggle/dataset-metadata.json /tmp/qaoa-data/
kaggle datasets create -p /tmp/qaoa-data

# 2. push and run the notebook (GPU accelerator is set in kernel-metadata.json)
kaggle kernels push -p kaggle
kaggle kernels status USERNAME/qaoa-angle-baseline
kaggle kernels output USERNAME/qaoa-angle-baseline -p out/     # submission.csv, baseline_mlp.pt
```

`enable_internet` is `false` — nothing is downloaded at runtime.

All four notebooks run in this same environment: same dataset (`J.npy`, `h_train.npy`,
`QAOA.py`), same GPU kernel, no internet, and nothing beyond numpy/torch/matplotlib. The kaggle
CLI reads exactly one `kaggle/kernel-metadata.json`, so to push a different notebook edit its two
identifying fields and push again:

```jsonc
"id":        "USERNAME/qaoa-massive-multistart",              // must be unique per kernel
"code_file": "../notebooks/03_massive_multistart.ipynb",
```

Notebook 03 adds one import over notebook 02 — `torch.quasirandom.SobolEngine`, part of torch
since 1.0, so no extra dependency. Its GPU peak is about the same: the Adam stage uses the same
2048-row chunks as notebook 02 (~4 GiB), and the screening stage adds ~1.75 GiB at
`screen_rows=16384`, both comfortable on a 16 GiB T4 or P100.

## Cost control

`CFG` in cell 2 is the only knob. Before the expensive run, the notebook times **one full-size
chunk** and prints a projected wall-clock for the whole job, so you can interrupt and lower the
budget rather than find out an hour in.

```python
CFG = dict(n_new=8000, n_restarts=24, steps=350, polish_steps=400, lr=0.06, rows_per_chunk=8192)
QUICK = False   # True -> ~2 min smoke run, verifies the notebook end-to-end
```

Cost scales as `n_total * (n_restarts * steps + polish_steps)`. `rows_per_chunk` bounds GPU
memory at roughly `rows * 200 KiB` (~1.6 GiB at 8192); lower it if you hit OOM.

Labels are cached to `labels_main.npz` in the working directory, so re-running the notebook
after generation is a couple of minutes. Download that file from the kernel output to avoid
paying for generation twice.

## Outputs

- `submission.csv` — `id, gamma_0..gamma_4, beta_0..beta_4`, one row per input instance
- `baseline_mlp.pt` — trained weights plus the feature scaler
- `labels_main.npz` — the generated dataset (`h`, `gamma`, `beta`, `p_ground`, `is_train_pool`)

## Massive multistart (notebook 03)

Same three organiser files, same `find_file()` discovery, same Kaggle push flow — only the
notebook name changes. It runs a three-stage funnel per instance:

1. **screen** ~65k scrambled-Sobol angle vectors with a forward-only pass (no autograd, so ~3×
   cheaper than a gradient step) and keep the best 256;
2. **coarse** 60 Adam steps on those, keep the best 32;
3. **fine** 400 Adam steps on those, keep the winner.

Every stage keeps the better of (refined point, incoming point), so a stage can never score worse
than its input.

```python
CFG = dict(n_sobol=65536, n_ramp=2048, keep_screen=256, steps_coarse=60,
           keep_coarse=32, steps_fine=400, lr=0.08, lr_fine=0.05,
           elite_from=64, elite_per=4, screen_rows=16384, adam_rows=2048)
QUICK = False   # True -> tiny end-to-end smoke run
```

`n_sobol` is the main quality knob and the dominant cost — scale it to fill the session. The
config cell prints the work split in gradient-equivalent units before anything runs; expect
roughly 15–20 min on a T4 at the defaults. `screen_rows`/`adam_rows` bound GPU memory only
(≈1.6 GiB and ≈4.4 GiB respectively at the defaults); halve on OOM.

Two things it measures rather than assumes:

- **§4** re-runs notebook 02's method (uniform random restarts) at a matched Adam budget and
  prints the ratio. On a small CPU smoke run the screened funnel scored 0.230 against 0.125 for
  random restarts — if that ratio ever drops below 1, the screen is not paying for itself and the
  budget belongs in `steps_fine`.
- **§7** projects the 500-instance wall clock against the 600 s inference limit. At full breadth
  it does not fit, which is expected: this is a search, not a model. Use it as a reference score
  and as a label/candidate generator (`multistart_angles.npz`), or shrink `n_sobol` as §7
  suggests for a legal safety submission.

Outputs: `submission.csv` and `multistart_angles.npz` (`h`, `gamma`, `beta`, `p_ground`).

## CMA-ES with restarts (notebook 04)

Same three organiser files, same `find_file()` discovery, same Kaggle push flow. It adds no
imports over notebook 03 — the evolution strategy is written out in the notebook (~120 lines of
torch), because the Kaggle kernel has `enable_internet: false` and the `cma` package cannot be
installed.

`runs x 500` independent CMA-ES instances are advanced in lockstep in the same CUDA kernels, so
covariance algebra costs nothing next to the simulator calls. Restarts happen at two scales: a run
that trips one of Hansen's termination criteria has its GPU slot reinitialised on the spot
(a `p_elite` share of those reseeded from angles that won on *other* instances), and each **IPOP
wave** doubles the population size.

```python
CFG = dict(cma_evals=90_000, waves=2, runs=8, lam=16, sigma0=0.40,
           tolfun=1e-11, tolx=1e-11, stag_gens=100, hist_len=20, p_elite=0.25,
           init=(0.50, 0.25, 0.25), n_screen=4096,
           keep=16, polish=250, polish_lr=0.05,
           eval_rows=16384, adam_rows=2048, eig_every=None)
QUICK = False   # True -> truncates h_train to 24 instances for an end-to-end smoke run
```

The budget is stated once, in **forward-eval-equivalents per instance** (a gradient step ≈ 3
forward evals), so it is directly comparable with notebook 03: the defaults spend ~110k against
notebook 03's ~152k, and §1 prints the split before anything runs. `cma_evals` is the main quality
knob. `eval_rows`/`adam_rows` bound GPU memory exactly as in notebook 03 (≈1.6 GiB and ≈4.4 GiB);
halve on OOM. Raise `eig_every` to 3–5 if batched `torch.linalg.eigh` turns out slow on the kernel.

Four things it measures rather than assumes:

- **§3** runs the strategy on sphere, a `1e6`-conditioned ellipsoid, Rosenbrock and Rastrigin
  before touching QAOA, and asserts on the results. A subtly broken CMA-ES still looks like it is
  optimising; these four catch it. Rastrigin also demonstrates the population-size effect the IPOP
  waves are buying (λ=10 stalls at f≈6, λ=200 reaches 0).
- **§7** is the head-to-head at a matched budget: Adam multistart (nb 02), Sobol funnel (nb 03),
  CMA-ES alone, CMA-ES + polish, and CMA-ES with eager restarts. CPU probes during development
  found a **budget crossover** — at ~11.5k evals/instance Adam multistart wins (0.233 vs 0.217),
  at ~24k CMA-ES wins (0.274 vs 0.242) — and that **tightening the restart triggers costs ~10%**
  (0.247 vs 0.274), because runs get killed while still improving. Both are 6–8 instances on a
  CPU and are indicative only; §7 re-measures on the GPU at the budget actually in use.
- **§8** reports which seeding family produced each instance's winning angles, which restart
  criterion is ending runs, and what the second wave bought — the three numbers that tell you how
  to retune `init`, `stag_gens` and `waves`.
- **§9** projects the 500-instance wall clock against the 600 s inference limit, as notebook 03 does.

Outputs: `submission.csv`, `cma_angles.npz` (`h`, `gamma`, `beta`, `p_ground`), plus
`cma_history.csv` (one row per generation) and `cma_meta.json` (config, timings, restart
statistics, score) so runs stay comparable across configs.

## Reading the results

The evaluation table is the point of the notebook. `best CONSTANT angles` is the number that
matters most: constant submissions score 0 by the rules, so a model that does not clearly beat it
has not learned anything about `h`. `per-instance oracle` is the label quality — a hard ceiling on
any model trained against those labels.
