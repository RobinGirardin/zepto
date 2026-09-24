# Empirical validity of Zepto cost estimates (Apertus)

This document is the pre-registered experimental design for a validity study of
Zepto’s analytical peak video-memory (VRAM) and floating-point operation (FLOP)
estimates. The comparison target is an executable HuggingFace / PyTorch replica
of the same graph, called the **twin**. The study is restricted to the Apertus
family (`model_id = apertus`). Families differ in architecture and in the
options they expose, so estimates are not comparable across families. The same
protocol is intended to be reused later for other families without changing the
hypotheses, outcomes, or analysis.

The study is an **equivalence** test: it asks whether the mean disagreement
between Zepto and the twin lies inside a pre-registered practical margin, not
whether a difference can be detected.

The estimand is the mean relative error on the **executable Apertus-like grid**
defined in Section 7 (architectures and workloads that fit a single study GPU
of about 16 GB). It is not the mean error over production Apertus 8B or 70B
serving configurations.

## 1. Scope

Apertus architecture type is fixed: eager grouped-query attention, xIELU,
llama3 rotary embeddings, and fused cross-entropy on the training forward
pass. Those mechanisms are not experimentally crossed.

The twin is a randomly initialized HuggingFace `ApertusForCausalLM` with the
same sampled options. No checkpoint weights are loaded. Weight values do not
enter the memory or FLOP estimates, except through the chosen numeric
precision.

The experiment requires a CUDA backend because the allocated-memory counters
used as the VRAM probe are only available there. Attention runs with the
eager backend so that fused kernels such as FlashAttention or scaled-dot-product
attention do not change the live set. Precision is uniform 32-bit or 16-bit
floating point (`fp32` or `fp16`). Mixed precision is out of scope because
Zepto does not model it. Key-value caching is disabled (`use_cache=False`)
and gradient checkpointing is off. Rotary tables and attention masks are
sized to the scored window `max(sequence_length, 32)` on both Zepto and the
twin.

Batch semantics are one parallel pass of batch size \(B\) and sequence length
\(S\). In Zepto this is `micro_batches = 1`. Sequential micro-batch summation
is out of scope.

The study does not evaluate production 8B or 70B serving, other model
families, fused-kernel optimization, or hardware other than the recorded GPU.

## 2. Hypotheses

A conventional test of mean difference treats “no difference” as the null and
is therefore the wrong tool when the scientific claim is that two methods
agree. This study uses **two one-sided tests** (TOST; Schuirmann 1987; Lakens
2017). Equivalence is declared only when the mean lies inside a pre-registered
interval around zero.

Let \(y\) be the Zepto estimate and \(t\) the twin probe for the same
hypothesis. The **relative error** is

$$
\mathrm{RE} = \frac{y - t}{t}.
$$

The equivalence margin is \(\Delta = 0.10\) (relative error in
\([-10\%, +10\%]\)). This margin is a planning tolerance, not a claim that
ten percent is the smallest difference that would matter for every GPU.

**H1 (VRAM).** On the executable Apertus-like grid, for the same architecture,
sequence length, batch size, execution phase, and precision, the mean of

$$
\mathrm{RE}_{\mathrm{vram}} = \frac{y_{\mathrm{vram}} - t_{\mathrm{vram}}}{t_{\mathrm{vram}}}
$$

is equivalent to 0 within \(\pm 0.10\).

**H2 (FLOPs).** On the same grid and the same matched conditions, the mean of

$$
\mathrm{RE}_{\mathrm{flop}} = \frac{y_{\mathrm{flop}} - t_{\mathrm{flop}}}{t_{\mathrm{flop}}}
$$

is equivalent to 0 within \(\pm 0.10\).

H2 is a test of **matrix-multiply and convolution ledger parity**, not of
Zepto’s full planning FLOP total. The twin probe is PyTorch’s
`FlopCounterMode`, which counts only a registered set of general
matrix-multiplies and convolutions and does not count the element-wise
arithmetic of an AdamW parameter update. Zepto’s corresponding column,
`y_flop_fcm`, applies the same allow-list and also excludes the update.
Those omissions are a pre-registered limitation of the probe, not a secondary
analysis.

H1 and H2 are each tested in every combination of execution phase
(inference, training) and precision (`fp16`, `fp32`). That yields **eight**
primary TOSTs. Phase and precision are within-subject factors: every subject
is measured in both phases and both precisions. The observation in each TOST
is that subject’s relative error in that combination. The two precisions are
not averaged.

Each TOST uses \(\alpha = 0.05\) for its pair of one-sided tests. Those two
tests are not Bonferroni-adjusted: they already form a single intersection-union
test. Family validity is a **conjunction**: the claim that Zepto is valid on
this protocol holds only if all eight TOSTs declare equivalence. If any
combination misses the margin, or is inconclusive because the confidence
interval overlaps a bound, the family claim fails. An inconclusive TOST is
not evidence that Zepto is inaccurate to ten percent.

A separate test of whether relative error differs by phase or precision
(\(\mathrm{RE} \approx \mathrm{phase} \times \mathrm{precision}\)) is
secondary and is not part of the conjunction.

## 3. Constructs and locked column pairs

### 3.1 VRAM (H1)

Zepto’s \(y_{\mathrm{vram}}\) is the analytical peak of live bytes. It includes
parameters, activations, graph-local workspace, and a CUDA cuBLAS runtime
workspace. A **cuBLAS handle** is a persistent library context. PyTorch
allocates a workspace buffer per handle; the size per handle is the formula
in PyTorch’s `parseChosenWorkspaceSize()` (about 8.5 MiB before compute
capability 9.0, and 32 MiB from 9.0 onward). Inference and a training
forward pass bill **one** handle. A training backward pass bills a second
handle, so a full training sequence bills **two**. The optimizer step bills
**zero**. Zepto uses the same counts. When a training horizon contains
forward, backward, and optimizer steps, the billed workspace is the
**maximum** number of handles live at once, not the sum across steps.

The twin probe \(t_{\mathrm{vram}}\) is
`torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` and a
device synchronize. That value is the caching-allocator peak while the
process is live. It includes tensors already live at the start of the window
(parameters, and after a training warm-up also gradients and optimizer
state) and leftover process state, of which the persistent cuBLAS workspaces
are the largest known term.

cuBLAS workspaces are created on the first general matrix-multiply that needs
them and are kept for the life of the Python process. `torch.cuda.empty_cache()`
and deleting the module do not free them. After any backward pass, two
handles typically remain. Inference then records two handles while Zepto
bills one.

Notably, HuggingFace AdamW optimizer, used for this study, is the `foreach` method as default. This method grep all the weights to udpate at the same time instead of looping through all of them. This is a gain of speed that occurs a slight memory inefficiency. Zepto AdamW policy will also settle with this default mechanism.

The study therefore warms the process **once**, before any scored row, with
a dummy forward and backward matrix-multiply so that two handles are already
live. Every later inference peak is adjusted by subtracting **one** handle
at the PyTorch / Zepto per-handle size, including the first subject and that
subject’s second precision. Training peaks are not adjusted. The adjusted
inference value is `target_vram`; the unadjusted allocator peak is
`target_vram_raw`. After this warm-up, any remaining leftover (allocator
alignment, other library workspaces) is treated as constant and is recorded
as a limitation. Process isolation in a new interpreter would reset the
handle pool but is not used on the Colab runtime.

`max_memory_allocated` is an allocator peak. It is not reserved device
memory and is not a “will this job fit” planning quantity. H1 tests
agreement with that allocator peak.

### 3.2 FLOPs (H2)

FLOP counts are obtained by applying a counting convention to a graph, not
by reading hardware performance counters. Zepto and `FlopCounterMode` both
use two FLOPs per multiply-add: a scalar operation counts as one FLOP, and
a product of \(A \in \mathbb{R}^{m \times n}\) with \(B \in \mathbb{R}^{n \times k}\)
counts as \(2mnk\).

Zepto’s full total (`y_flop`) also bills operations that
`FlopCounterMode` does not: the counter’s registry is limited to
`matmul`, `bmm`, `addmm`, `conv2d`, and `conv3d`, and it does not bill
AdamW element-wise updates. The primary Zepto column for H2 is therefore
`y_flop_fcm`, which restricts Zepto to that registry and excludes the
update.

The twin column \(t_{\mathrm{flop}}\) is `target_flop` from
`FlopCounterMode` over the scored operations. Inference is one forward
pass. Training is forward, backward, and `optimizer.step()`. The optimizer
step is included so that the scored window matches the named training step.
Neither ledger counts the update, so including the step does not change the
H2 totals. It can still move the VRAM peak through transient optimizer
scratch, which is why the same step is used for the training VRAM window.

Inference already records FLOPs and VRAM in two windows: the counter first,
then a reset peak window. Training does the same. The training VRAM peak is
**not** taken while `FlopCounterMode` is active, because the counter’s
dispatch mode allocates bookkeeping tensors and changes the live set.

## 4. Design

The design is a \(2 \times 2\) within-subject factorial.

| Factor | Levels | Assignment |
|--------|--------|------------|
| Phase | inference, training | Both measured on every subject |
| Precision | `fp32`, `fp16` | Both measured on the same architecture and the same \((B, S)\) |

A **subject** is the independent experimental unit: one Apertus option set
together with one batch size and one sequence length,
`(configuration_id, seq_len, batch_size)`. `configuration_id` identifies
the architecture only. Uniqueness for sample size \(n\) is the full subject
key, so two workloads on the same graph are two subjects. Each subject
contributes four relative-error observations per hypothesis (phase ×
precision) and therefore eight observations across H1 and H2.

Training step index is not a factor and is not extra \(n\). Zepto’s
analytical \(y\) does not change across scored steps; treating steps as
replicates would inflate power. The twin protocol is one unscored training
warm-up on that module (to instantiate optimizer state), then **K = 1**
scored training step.

The dependent variable is the relative error defined in Section 2. It
places inference and training, and both precisions, on a common scale.

## 5. Analysis

### 5.1 Primary: TOST on mean relative error

Each of the eight phase × precision combinations is tested with a one-sample
TOST on relative error (Schuirmann 1987; Lakens 2017):

- \(H_{01}\colon \mu \le -0.10\) versus \(H_{11}\colon \mu > -0.10\)
- \(H_{02}\colon \mu \ge +0.10\) versus \(H_{12}\colon \mu < +0.10\)

with \(\alpha = 0.05\). Equivalence in that combination is declared only if
both one-sided tests reject. The eight tests are not Bonferroni-adjusted.
The conjunction in Section 2 already keeps the probability of a false
overall pass at no more than \(\alpha\).

Each TOST receives one relative error per completed subject in that
combination.

Relative error is asymmetric: the same absolute gap yields a smaller
relative error when \(t\) is larger. The TOST bounds remain on this scale.
Bland–Altman plots (Section 5.2) show the signed difference against the
magnitude of the two measurements so that the asymmetry is visible.

### 5.2 Descriptive agreement: Bland–Altman

For each hypothesis and each phase × precision combination, plot

- horizontal axis: \(\frac{y + t}{2}\),
- vertical axis: signed difference \(y - t\).

Each panel reports the mean difference and the limits of agreement (Bland
and Altman 1986). These figures do not replace TOST.

### 5.3 Missing data

Builds that fail, including out-of-memory failures, are written to a ledger
and are not dropped silently. The primary analysis uses completed subjects
only. The ledger is the sampling-frame appendix.

## 6. Power and sample size

Sample size is obtained from a one-sample TOST power calculation in the R
package TOSTER (`power_t_TOST`, `type = "one.sample"`; Lakens 2017).

If each of the eight tests were powered to 0.95 and were independent, the
probability that all eight would pass would be \(0.95^{8} \approx 0.66\).
The tests share subjects, so they are not independent and the drop is
smaller, but it is still a loss of joint power. The study offsets that loss
by raising the **per-combination** power rather than weakening the
conjunction:

$$
0.95^{1/8} \approx 0.9936.
$$

`power_t_TOST` is therefore run with `power = 0.9936`. The significance
level stays \(\alpha = 0.05\); it is not divided by eight. The same \(n\)
is used for every combination. After the pilot, the largest of the eight
cell standard deviations is entered as \(\sigma\) and the call is solved
once.

| Input | Value |
|-------|--------|
| Estimand | mean relative error in one phase × precision combination (one value per subject) |
| Bounds | `eqb = 0.10` |
| Assumed true mean | \(\delta = 0\) |
| \(\alpha\) | 0.05 |
| Power \(1-\beta\) | 0.9936 per combination (conjunction of eight combinations targets joint power 0.95) |
| \(\sigma\) | residual standard deviation of subject-level relative error in that combination, from a protocol-matched pilot; use the largest of the eight cell values |

The resulting \(n\) is the number of **subjects** (distinct
`(configuration_id, seq_len, batch_size)`), not CUDA rows and not training
steps. Raising power from 0.95 to 0.9936 increases \(n\). That increase is
the planned offset for the conjunction.

A protocol-matched pilot is run first. It supplies \(\sigma\) and is also
used to inspect disagreement qualitatively before the main sample.

## 7. Sampling

Each subject is one valid Apertus option set together with one \((B, S)\)
pair that fits the study GPU. Independent integer draws on
`hidden_size`, `num_heads`, `head_dim`, and `intermediate_size` almost never
satisfy the identities of an Apertus graph. The study therefore draws a
small set of **free knobs** (quantities that may be chosen at random) and
computes the remaining dimensions from those knobs, so every accepted
architecture is valid by construction.

Free knobs are drawn uniformly from finite discrete sets. A uniform
distribution is used because the study does not posit a population
distribution over Apertus configurations and does not wish to overweight
particular widths or depths.

### 7.1 Architectural identities

The following identities must hold for every accepted architecture.

1. **Width.** Residual width is the product of query-head count and head
   width: \(\texttt{hidden\_size} = \texttt{num\_heads} \times \texttt{head\_dim}\).
2. **Grouped-query attention.** Query heads must partition evenly over
   key/value heads: \(\texttt{num\_heads} \bmod \texttt{num\_kv\_heads} = 0\).
3. **Even head width.** Rotary embeddings pair dimensions, so
   \(\texttt{head\_dim} \bmod 2 = 0\).
4. **Feed-forward width.** The expanded feed-forward width stays at an
   Apertus-like multiple of residual width, then aligns to 64:
   \(\texttt{intermediate\_size} = \mathrm{round}_{64}(\texttt{ffn\_mult} \times \texttt{hidden\_size})\).
5. **Positivity.** `hidden_size`, `intermediate_size`, `vocab_size`, and
   `num_layers` are all greater than zero.

### 7.2 Free knobs and derived options

| Identity | Free knobs | Derived later |
|----------|------------|---------------|
| Grouped-query grouping | `num_kv_heads`, `gqa_group` | `num_heads = num_kv_heads × gqa_group` |
| Width | even `head_dim`, with `num_heads` already derived | `hidden_size = num_heads × head_dim` |
| Even `head_dim` | `head_dim ∈ {64, 128}` | (already even) |
| Feed-forward expansion | `ffn_mult ∈ {4, 5.25}` | `intermediate_size = round_64(ffn_mult × hidden_size)` |
| Vocabulary | `vocab_size ∈ {1024, 4096}` | (already valid) |
| Depth | `num_layers ∈ {2, 4, 8}` | (already positive) |

Worked example. Draw `num_kv_heads = 4`, `gqa_group = 2`, `head_dim = 128`,
`ffn_mult = 5.25`, `num_layers = 4`, `vocab_size = 1024`. Then
`num_heads = 8`, `hidden_size = 1024`, and
`intermediate_size = round_{64}(5.25 × 1024) = 5376`. The option dictionary
passed to Zepto and to the twin is
`{hidden_size, intermediate_size, num_heads, num_kv_heads, num_layers, vocab_size, head_dim}`.
`gqa_group` and `ffn_mult` never appear on the twin; they only parameterize
the draw.

After this derivation, even `head_dim` and integer grouped-query groups are
identities, not rejection rules.

### 7.3 Architecture catalog

The free knobs are drawn from finite sets that keep 8B-like ratios while
remaining executable on the study GPU. The working catalog below is locked
after the protocol-matched pilot of Section 6.

- `gqa_group` \(\in \{1, 2, 4\}\)
- `num_kv_heads` \(\in \{2, 4, 8\}\)
- `head_dim` \(\in \{64, 128\}\)
- `num_layers` \(\in \{2, 4, 8\}\)
- `ffn_mult` \(\in \{4, 5.25\}\), with `intermediate_size` rounded to a multiple of 64
- `vocab_size` \(\in \{1024, 4096\}\) on the main frame; optional extra
  subjects at `131072` if they fit (embedding-dominated VRAM)

Subjects are drawn uniformly from the finite valid set generated by these
knobs. Independent continuous uniforms on wide integer ranges are not used.

### 7.4 Sequence length and batch size

Sequence length and batch size are not drawn from wide uniforms such as
`seq_len ~ U(1, 512)` and `batch_size ~ U(1, 64)`. Those tails produce
out-of-memory failures on a T4 or L4 and would silently change the
sampling frame.

Each subject receives a pair \((B, S)\) from a discrete catalog, or
uniformly from
\(\{(B, S) : B \ge 1,\ S \ge 32,\ B \cdot S \le T_{\max}\}\), where
\(T_{\max}\) is locked from the pilot together with a cap on estimated peak
memory versus device memory. The minimum sequence length of 32 matches the
windowed rotary context of Section 1. The same \((B, S)\) is reused for
both precisions.

### 7.5 ATEN operation

Zepto represent all the functions as the combination of their primitive operation,
For example, a function as $\text{softmax}$ is represented by its series of exponential, divide, exponential and sum operation. This allow to monitor backward propagation mechanism as the primitive operation level, and not per specific operation or layer. However, due to the wide-spread adoption certain functions such as the $\text{XIELU}$ activation, $\text{RMSNorm}$ or $\text{softmax}$, have their own C/C++ (aten) variant. Those function are the equivalent of a fused-kernel, a special series of operation that make the execution of those function faster but also cheaper on memory. To ensure comparability, Zepto twin will match PyTorch ATEN operation behavior for the availalbe function. This will be done by using a list of `region_implementaton_pins` for Zepto, that points the graph to use a fused-kernel version of ceratin region.

List of PyTorch ATEN operation that requires a dedicated zepto fused-kernel pins:

- `torch.rmsnorm`
- `torch.softmax`
- `XIELUActivation`
- `torch.SiLU`

### 7.6 Assignment

For each accepted subject, measurements follow the within-subject factorial
of phase and precision.

1. Before any scored row in the process, one dummy forward and backward
   matrix-multiply warms the cuBLAS handle pool (Section 3.1).
2. Zepto is evaluated once per precision on the shared options and
   \((B, S)\). Those estimates are analytical and do not consume a scored
   CUDA window.
3. The twin is built at that precision. Inference is measured first
   (FLOP window, then VRAM window). Training then uses one unscored module
   warm-up and **K = 1** scored step. The scored training step includes
   `optimizer.step()`. FLOPs and VRAM are recorded in separate windows.
4. The twin is torn down. Inference VRAM is stored as
   `target_vram = target_vram_raw −` one handle. Training VRAM is stored
   unadjusted.
5. The procedure is repeated at the other precision on the same options
   and the same \((B, S)\).

Precision is therefore not assigned by a coin flip across subjects.
Duplicate `(configuration_id, seq_len, batch_size)` keys are rejected so
that the \(n\) of Section 6 counts unique subjects.

## 8. Limitations

`FlopCounterMode` is the available executable FLOP probe. H2 therefore
tests agreement of two matrix-multiply and convolution ledgers. It does not
validate Zepto operations outside that registry or the AdamW update term.

After process warm-up, leftover bytes that are not the extra inference
handle are assumed constant. The study does not isolate each subject in a
new Python process.

The allocator peak is not reserved memory and is not a serving-fit claim.

The estimand is the executable grid of Section 7, not the Apertus 8B or 70B
serving distribution. Relative error from an additive leftover is larger on
small peaks than on large ones. The primary TOST still uses the relative
error of Section 2.

## Implementation

The executable pipeline for this protocol lives in
`studies/apertus-validity/`. Python collects subjects and writes CSV;
Quarto (`r/analysis.qmd`) runs the eight TOSTs and Bland–Altman panels.
See that directory’s README. `src/zepto/empirical/` is a measurement
harness, not the study.

A second instance of the same protocol for Granite lives in
`studies/granite-validity/` and is scoped in
`docs/framework-empirical-evaluation-granite.md`.

## Sources

```bibtex
@article{schuirmann1987tost,
  author  = {Schuirmann, Donald J.},
  title   = {A comparison of the two one-sided tests procedure and the power
             approach for assessing the equivalence of average bioavailability},
  journal = {Journal of Pharmacokinetics and Biopharmaceutics},
  year    = {1987},
  volume  = {15},
  number  = {6},
  pages   = {657--680},
  doi     = {10.1007/BF01068419}
}

@article{lakens2017equivalence,
  author  = {Lakens, Dani{\"e}l},
  title   = {Equivalence Tests: A Practical Primer for \(t\) Tests, Correlations,
             and Meta-Analyses},
  journal = {Social Psychological and Personality Science},
  year    = {2017},
  volume  = {8},
  number  = {4},
  pages   = {355--362},
  doi     = {10.1177/1948550617697177}
}

@article{blandaltman1986,
  author  = {Bland, J. Martin and Altman, Douglas G.},
  title   = {Statistical methods for assessing agreement between two methods
             of clinical measurement},
  journal = {The Lancet},
  year    = {1986},
  volume  = {327},
  number  = {8476},
  pages   = {307--310},
  doi     = {10.1016/S0140-6736(86)90837-8}
}

@manual{toster,
  title  = {{TOSTER}: Two One-Sided Tests ({TOST}) Equivalence Testing},
  author = {Lakens, Dani{\"e}l and Caldwell, Aaron},
  year   = {2024},
  url    = {https://CRAN.R-project.org/package=TOSTER},
  note   = {R package; use \texttt{power\_t\_TOST} for one-sample TOST sample size}
}
```
