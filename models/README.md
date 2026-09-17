# Shipped model artefacts

This directory holds what `solution.ipynb` loads to reproduce the submission **without
retraining**. It is part of the deliverable, unlike `runs/`, which holds training
artefacts and is gitignored.

| file | what it is |
|---|---|
| `moe_best.pt` | trained `src.moe.AngleMoE` — codebook + gate. Written by `docker compose run --rm moe` as `runs/moe/best.pt`; copy it here to ship it. |

`.gitignore` ignores `*.pt` everywhere and then un-ignores `models/*.pt`, so the
checkpoint commits normally while training checkpoints stay out. This README also exists
so the directory survives a clone — git does not track empty directories, and without it
`cp runs/moe/best.pt models/moe_best.pt` fails on a fresh checkout.

```bash
cp runs/moe/best.pt models/moe_best.pt
git add models/moe_best.pt && git commit -m "Ship the trained model" && git push
```
