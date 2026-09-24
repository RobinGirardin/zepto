# Apertus validity study

This directory is the executable study declared in
[`docs/framework-empirical-evaluation.md`](../../docs/framework-empirical-evaluation.md).
It is **not** `src/zepto/empirical/`. That package is a measurement harness.
This tree owns sampling, the failure ledger, and the pre-registered analysis.

The estimand is the mean relative error on the **executable Apertus-like grid**,
not production 8B or 70B serving.

| Hypothesis | Locked columns | Relative error |
|------------|----------------|----------------|
| **H1 VRAM** | `y_vram` vs `target_vram` | \((y - t) / t\) |
| **H2 FLOPs** | `y_flop_fcm` vs `target_flop` | \((y - t) / t\) |

Eight one-sample TOSTs (phase × precision × hypothesis), \(\Delta = 0.10\),
\(\alpha = 0.05\). Family validity is the **conjunction** of all eight.

A **subject** is `(configuration_id, seq_len, batch_size)`. Precision is not
sampled: every subject is measured in `fp16` and `fp32`. Training uses **K = 1**
scored step.

## Layout

```text
python/apertus_validity/   collection (catalog, sample, measure, ledger)
r/power.R                  sample size from the largest of eight cell SDs
r/analysis.qmd             eight TOSTs, conjunction, Bland–Altman, ledger
notebooks/                 Colab-capable collection
fixtures/                  tiny CSV so the Quarto doc renders without a GPU
artifacts/                 run outputs (gitignored)
```

## How to run

Python collects. Quarto analyzes. From the repo root:

```bash
# 1. Pilot: draw subjects and measure (needs CUDA)
python -m apertus_validity run \
  --n 12 --seed 1 \
  --out studies/apertus-validity/artifacts/pilot

# 2. Lock n from the largest cell SD (needs R + TOSTER)
Rscript studies/apertus-validity/r/power.R \
  --evaluation studies/apertus-validity/artifacts/pilot/evaluation.csv \
  --out studies/apertus-validity/artifacts/pilot/sample_size.json

# 3. Main sample (use n from sample_size.json)
python -m apertus_validity run \
  --n N --seed 2 \
  --out studies/apertus-validity/artifacts/main

# 4. Primary analysis
quarto render studies/apertus-validity/r/analysis.qmd \
  -P evaluation:../artifacts/main/evaluation.csv \
  -P ledger:../artifacts/main/ledger.csv
```

Render the fixture copy of the analysis without a GPU:

```bash
quarto render studies/apertus-validity/r/analysis.qmd
```

`pythonpath` for the study package is
`studies/apertus-validity/python` (already on pytest’s path).

## What this tree does not do

- It does not treat extra training steps as extra \(n\).
- It does not drop out-of-memory rows. Those go to `ledger.csv`.
- It does not test `y_flop` (full Zepto total) as H2. That column is diagnostic.
- It does not reuse `notebooks/apertus_empirical.ipynb`. That notebook is stale.
