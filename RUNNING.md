# Running the baseline

The deliverable is a single self-contained notebook: `notebooks/01_baseline_mlp.ipynb`.
It has no repo dependency — no clone, no `pip install` — so `kaggle kernels push` only needs the
notebook itself. It needs a GPU; on CPU the label-generation step is roughly 80× slower.

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

## Reading the results

The evaluation table is the point of the notebook. `best CONSTANT angles` is the number that
matters most: constant submissions score 0 by the rules, so a model that does not clearly beat it
has not learned anything about `h`. `per-instance oracle` is the label quality — a hard ceiling on
any model trained against those labels.
