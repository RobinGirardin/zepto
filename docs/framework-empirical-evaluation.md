# Empirical validity of Zepto cost estimates (Apertus)

This document is the pre-registered experimental design for a validity study of
Zepto’s analytical **peak VRAM** and **FLOP** estimates against an executable
HuggingFace / PyTorch twin. The claim is scoped to the **Apertus** family. Reason for this scope is the incomparability between model family due to their different architecture and configuration options. The same protocol is intended to be reused later for other families without changing the hypotheses, outcomes, or analysis.

In order to validate Zepto's estimates, we perform an **equivalence** study that seeks to determine that Zepto's estimation are equivlent to HuggingFace / PyTorch twin.

## 1. Scope

- **Family:** Apertus only (`model_id = apertus`). Architecture type (eager GQA,
  xIELU, llama3 RoPE, fused CE on train) is fixed. Those mechanisms are not
  experimentally crossed.
- **Twin:** random-init HuggingFace `ApertusForCausalLM` matching the sampled
  architecture options. No checkpoint weights. Weight, except for their precision level, are irrelevant for the study as they do not influence memory and computational costs estimates.
- **Context:** The experiment requires a CUDA backend because the necessary allocated memory monitoring PyTorch tools are only available on said backend. We use the `attention_backend="eager"` to zero-out the potential influence of kernel optimization. For precision-level, we use uniform **fp32** and **fp16** level, with no mixed-precision, simpy because Zepto does not support it. We set `use_cache=False` and no
  gradient checkpointing. For RoPE context length, we use a windowed approach for both Zepto and HuggingFace / PyTorch, setting `max(sequence_length, 32)`.
- **Batch semantics:** one parallel `(B, S)` pass for both inference and training schemes. Structurally, this assumes a `micro_batches=1` in Zepto. Sequential
  micro-sum is out of scope.
- **Out-of-scope:** This study does not evaluate production 8B/70B serving, other families estimates, fused kernel optimization mechanism such as FlashAttention or SDPA backends, or hardware other than the recorded GPU.

## 2. Hypotheses

Planning validity means the mean relative error of Zepto versus the probe lies
inside a pre-registered practical margin. Regular statistical approach testing for mean differences assumes a null hypothesis where there is no difference between groups, thus testing for the alternative hypothesis where there is indeed a difference. This approach is unsuited for this study as we wish to test for the alternative hypothesis where there is no difference between group means instead.
Let the relative error be expressed as

$$
\[
\mathrm{RE} = \frac{y - t}{t}
\]
$$

where $y$ is the Zepto estimate and $t$ is the documented probe for that
hypothesis. Equivalence bounds are arbitrarily set to $\Delta = 0.10$ (relative error in $([-10\%, +10\%])$).

**H1 (VRAM).** For Apertus, on the same architecture, sequence length, batch
size, phase, and precision, the mean of

$$
\mathrm{RE}_{\mathrm{vram}} = \frac{y_{\mathrm{vram}} - t_{\mathrm{vram}}}{t_{\mathrm{vram}}}
$$

is equivalent to 0 within $\pm 0.10$.

**H2 (FLOPs).** For Apertus, on the same architecture, sequence length, batch
size, phase, and precision, the mean of

$
\mathrm{RE}_{\mathrm{flop}} = \frac{y_{\mathrm{flop}} - t_{\mathrm{flop}}}{t_{\mathrm{flop}}}
$

is equivalent to 0 within $\pm 0.10$.

H1 and H2 are each tested in every phase × precision cell (inference/training
× fp16/fp32). That is **8** primary TOSTs. Phase and precision are within-subject
factors: every subject is measured in both phases and both precisions. The
observation in each TOST is that subject's RE in **that cell**, not an average
across precisions.

Each TOST uses \(\alpha = 0.05\) for its two one-sided tests; those two tests
are not Bonferroni-split. Family validity is a **conjunction**: the claim holds
only if all eight TOSTs declare equivalence. A miss or an inconclusive cell
(the confidence interval overlaps a bound) means the family claim fails. An
inconclusive TOST is not evidence that Zepto is inaccurate to 10%. The
`phase × precision` difference test is secondary and outside this family.

## 3. Constructs and locked column pairs

### 3.1 VRAM (H1)

**Zepto y\):** `y_vram` — full analytical peak memory, which includes parameters,
activations, graph workspace, and the CUDA cuBLAS runtime workspace policy. The latter being measured by (`peak_live_bytes` on inference and `peak_vram` on the training horizon.
HuggingFace / PyTorch Inference and training forward bills **1** cuBLAS handle, while training backward bills an additional handle ¬ thus **2** for a full training sequence. The optimizer step bills **0** cuBLAS workspace. Zepto follow this standards as well; **1** handle for inference or forward pass and **2** for full training or backward pass. An horizon of `forward + backward + optimizer` steps would thus bills `1 + 2 + 0` cublas workspace. To avoid overcounting, zorizon `runtime_workspace` is the **maximum** of billed handles across all steps, not the sum of handles across steps.

**PyTorch probe.** We measure allocated memory at a given time step through **`torch.cuda.max_memory_allocated()`** after `reset_peak_memory_stats()`
and a synchronize. That value is the caching-allocator peak while the process
is live. it includes

- tensors already live at the start of the window (parameters, and
  optimizer state and gradients after warmup),
- leftover process state from earlier draws (typically persistent cuBLAS
  workspace),

More specifically, cuBLAS workspaces are instantiated once during the forward and backward pass, and are re-used throughout the whole study across all samples. This means that, after the first sample measurements, 2 cuBLAS workspace handles sits in memory. As a consequence, the second and further samples will records consistently 2 workspace handles across all phases. In the inference phase of the second sample onward, this introduce a bias of a full cuBLAS workspace handles. The study adjust for this issue by substracting a full cuBLAS workspace handle on all inference measurement after the first draw. The adjusted memory count is represented by the `target_vram` variable, while the unadjusted is represented by `target_vram_raw`.

### 3.2 FLOPs (H2)

**Zepto $y$ (`y_flop_fcm`):** FLOP accounting is done theoretically. Measuring FLOP count accurately is extremely hard to do practically because it involves hardware instruction monitoring and specific support for different hardware. As such, FLOP is generally estimation **theoretically**. This involves its own set of assumption, such as wich convention is used to calcule FLOP per operation and which operation is accoutned for.
Zepto use the 2 FLOP per multiply-accumulate convention, as most in the machine learning fields. In this convention, scalar operation are counted as 1 FLOP and matrix multiplication, say between $A \in \mathff{R}^{m, n}$ and $B \in \mathff{R}^{n, k}$ are computed through the $2mnk$ rule. Although Zepto and `FlopCounterMode`, the PyTorch probe, use the same counting convention, Zepto account for more operation that `FlopCoutnerMode`, which is limited to matrix multplication `matmul`, `bmm`, `addmm` and convolution `conv2d` and `conv3d`. Lastly, `FlopCounterMode` does account for the FLOP created by weight updates. A `flop_counter_mode` Zepto policy restrict Zepto's FLOP accounting to only the operation supported by `FlopCounterMode` and without weight update FLOP to ensure parity and comparability.

**PyTorch $t_flop$:** `target_flop` — `torch.utils.flop_counter.FlopCounterMode`
over the scored ops (inference: one forward; training: forward + backward +
`opt.step()`).

Inference FLOPs and VRAM are measured on a single CUDA forwards, flop counter and vram peak in a single window. Training uses an unscored warmup step, then one scored step for both FLOP and VRAM. Those windows are not a single paired kernel trace.

## 4. Design

The design is a  2 × 2 **within-subject** design.

| Factor | Levels | How it is assigned |
|--------|--------|--------------------|
| Phase | inference, training | Both measured on every draw |
| Precision | fp32, fp16 | both precisions measured on the same sample `(architecture, B, S)` |

A **subject** (independent experimental unit) is a set of configuration option for for a single model family, alongside with sequence length and batch size `(configuration_id, seq_len, batch_size)`. Combined, they form the full model architecture shared between Zepto and HuggingFace / PyTorch twin. Thus, each subject contributes four RE observations per hypothesis (phase × precision).

**Training `step` is not a factor and not extra $N$.** Zepto’s analytical
$y$ is constant across scored steps; only the CUDA probe can drift. Using
steps as additional samples inflates power. As such,  we use the following protocol: one unscored warmup step (HF only), then **K = 1** scored training step.

**Dependant Variable** The dependant variable is the relative error between Zepto's and HuggingFace / PyTorch probe. This allow a standardized measure across phase and precision level condition will have different value scale.

## 5. Analysis

### 5.1 Primary: TOST on mean relative error

H1 and H2 are each tested in every phase × precision cell (Schuirmann 1987;
Lakens 2017): eight one-sample TOSTs on $\mathrm{RE}$ against
$\Delta_L = -0.10$, $\Delta_U = +0.10$, $\alpha = 0.05$.

- $H_{01}\colon \mu \le -0.10$ vs $H_{11}\colon \mu > -0.10$
- $H_{02}\colon \mu \ge +0.10$ vs $H_{12}\colon \mu < +0.10$

We declare equivalence in a cell only if **both** one-sided tests reject. Those
two tests already form an intersection-union test, so they share one
$\alpha = 0.05$ and are not Bonferroni-split. Across the eight cells we also
do not split $\alpha$: family validity is the conjunction in Section 2, which
keeps the false overall pass at \(\le \alpha\) without a Bonferroni correction.
The observation fed to each TOST is that subject's RE in that cell (one row
per subject per cell). Do not average fp16 and fp32.

The `phase × precision` difference test remains secondary:
$\mathrm{RE} \approx \text{phase} \times \text{precision}$.

Although the relative error is easy to read, it is asymmetric; for a similar absolute difference, a bigger denominator will give a smaller RE.
The TOST bounds are nevertheless defined on this scale. Bland–Altman
plots (below) also show the signed difference against the probe magnitude so
the asymmetry is visible.

### 5.2 Descriptive agreement: Bland–Altman

For each hypothesis and each phase × precision cell, plot

- x: both probe means $\frac{y+t}{2}$,
- y: signed difference $y - t$.

This graph report mean difference and the limits of agreement, facetted by phase and precision. These figures do not replace TOST, it only give descriptive information on the two measurement equivalence.

### 5.3 Missing data

- OOM or other failed builds are recorded in a ledger and are **not** silent
  drops. Primary analysis uses completed subjects only; the ledger is the
  sampling-frame appendix.

## 6. Power and sample size

To estimate the necessary sample size, we perform a power-analysis using R `TOSTER`package (Lakens 2017; TOSTER `power_t_TOST`, `type = "one.sample"`) as a primary one-sample TOST.

A per-cell power of 0.95 would not give a 0.95 chance that all eight TOSTs
pass. If the cells were independent, that joint power would be
\(0.95^{8} \approx 0.66\). Cells share subjects, so the drop is smaller, but
still real. We offset that conjunction loss by raising **per-cell** power
rather than adding a weaker-joint-claim limitation:

\[
0.95^{1/8} \approx 0.9936
\]

`power_t_TOST` is therefore run with `power = 0.9936`. \(\alpha\), `eqb`, and
`type` stay as below; we do not Bonferroni-split \(\alpha\). The same \(n\)
is used for every cell. After the pilot, take the **largest** cell-level
\(\sigma\) (the hardest cell) and solve once.

| Input | Value |
|-------|--------|
| Estimand | mean $\mathrm{RE}$ in one phase × precision cell (one RE per subject in that cell) |
| Bounds | $\mathrm{eqb} = 0.10$ |
| Assumed true mean | $\delta = 0$ |
| $\alpha$ | 0.05 |
| Power $1-\beta$ | 0.9936 (per cell; conjunction of 8 cells targets joint power 0.95) |
| $\sigma$ | residual SD of subject-level RE in that cell — **from a protocol-matched pilot**; use the largest of the eight cell SDs |

The analysis provide us with the required $n$, or the number of **subjects** (distinct `(configuration, B, S)`), not CUDA rows, not training steps. Raising power from 0.95 to 0.9936 increases \(n\) (typically on the order of 20–35% once \(\sigma\) is not tiny). That extra sample is the planned offset for the conjunction.

**Pilot before the main N.** To determine the $\sigma$ for the power analysis, we perform a pilot study, following the same structure as the main one. This pilot study will allow us to check the difference between Zepto and the probe qualitatively, before going for an extensive quantitative analysis.

## 7. Sampling

The estimand is the mean relative error over the Apertus family defined in Section 1. As previously mentioned, each sample if a set of configuration options alongside a distinct batch size and sequence length. To limit costs, the HuggingFace / PyTorch configuration, sequence length and batch size should fit a T4 / L4 chip, with roughly 16 GB of VRAM. Moreover, certain configuration values are bounded by constraints. For example, the number of hidden dimension, commonly called $d$, is usually proportional to the amount of attention heads $h$ and those head hidden dimensions $d_h$, so that $d = h \times d_h$. As a consequence, all configuration values and combination cannot be randomly generated. To paliate this, we select a set of configuration values that can be randomly generated (called *free knobs*) and derive the other from them.

The sampling of those *free knobs* involves a uniform distribution of integers (thus discretetized.). The reason behing the choice of a uniform distribution is that we have no prior knowledge about those configuration values distribution, and thus we would like to avoid any a-priori by attributing more weight to particular values.

### 7.1 Free Knobs and Sampling Constraints

Each architecture is obtained by sampling a set of *free knobs* and deriving the remaining dimensions, so that every accepted draw satisfies the Apertus sample by construction. The *free knobs* are determined based on model constraints. Find below a list of those constraints and the list of chosen randomly initialized values, as well a their derivatives.

### 7.1.1 Constraints

These are the constraints that independent integer draws would keep violating.

1. **Width identity.** Hidden dimension length, or sequence embeddings dimension length, is the product of query heads and head dimension length:
   $$
   \text{hidden_size} = \text{num_heads} \times \textthead_dim}.
   $$
   You cannot pick `hidden_size`, `num_heads`, and `head_dim` independently.

2. **GQA grouping.** Query heads must partition evenly over key/value heads:
   $$
   \text{num_heads} \bmod \text{num_kv_heads} = 0.
   $$
   HuggingFace and Zepto both reject a non-integer group size (for example 7 query heads and 4 KV heads).

3. **Even `head_dim` (RoPE).** Rotary positional embedding pair dimensions, so
   $$
   \text{head_dim} \bmod 2 = 0.
   $$

4. **FFN intermediate dimension length.** `intermediate_size` is the expanded FFN hidden dimension length (width of the second FFN weight matrix). It must be positive, and for this study it is meant to stay at an Apertus-like expansion of `hidden_size`, then aligned:
   $$
   \text{intermediate_size} \approx \mathrm{round}_{64}(\text{ffn_mult} \times \text{hidden_size}).
   $$
   Drawing `intermediate_size` from a wide integer range independently of `hidden_size` produces arbitrary, non-Apertus FFN.

5. **Positivity.** `hidden_size`, `intermediate_size`, `vocab_size`, and `num_layers` must all be $> 0$.

### 7.1.2 Free Knob

A **free knob** is a quantity we may draw at random. Everything else is a function of those draws, so a valid option dict is produced without rejection on the identities above.

| Constraint | Why independent draws fail | Free knobs | Derived later |
|------------|----------------------------|------------|---------------|
| GQA grouping | `num_heads` and `num_kv_heads` from separate uniforms often fail `num_heads % num_kv_heads == 0` | `num_kv_heads`, `gqa_group` | `num_heads = num_kv_heads * gqa_group` |
| Width identity | `hidden_size` drawn on its own will not equal `num_heads * head_dim` | `head_dim` (from an even set) plus the derived `num_heads` | `hidden_size = num_heads * head_dim` |
| Even `head_dim` | `Uniform` over integers includes odds; RoPE then fails | `head_dim ∈ {64, 128}` | none; evenness is already in the support |
| FFN expansion | `intermediate_size` independent of `hidden_size` is not an Apertus MLP | `ffn_mult ∈ {4, 5.25}` | `intermediate_size = round_64(ffn_mult * hidden_size)` |
| Positivity / discrete vocab | `vocab_size ~ U(1, 131072)` is almost never an Apertus table and often OOMs | `vocab_size ∈ {1024, 4096}` (optional 131072) | none; the set is already valid |
| Depth | no identity, only positivity | `num_layers ∈ {2, 4, 8}` | none |

As a working example, a random sample could draw the following values.

- `num_kv_heads = 4`
- `gqa_group = 2`
- `head_dim = 128`
- `ffn_mult = 5.25`
- `num_layers = 4`
- `vocab_size = 1024`

then derive

- `num_heads = 4 × 2 = 8`
- `hidden_size = 8 × 128 = 1024`
- `intermediate_size = round_{64}(5.25 × 1024) = 5376`

The option dict handed to Zepto and HuggingFace is then exactly `{hidden_size, intermediate_size, num_heads, num_kv_heads, num_layers, vocab_size, head_dim}`. `gqa_group` and `ffn_mult` never appear on the twin; they only parameterize the draw.

That is the sense in which the knobs “solve” the constraints: **sample the generators, compute the dependents.** After that, `head_dim % 2 == 0` and `num_heads % num_kv_heads == 0` are identities, not rejection rules.

### 7.2 Sampling set

The free knobs are drawn from finite discrete sets that preserve 8B ratios while remaining executable on a single study GPU. The sets below are the working catalog;
they will be locked after the protocol-matched pilot of Section 6.

- `gqa_group` $\in \{1, 2, 4\}$
- `num_kv_heads` $\in \{2, 4, 8\}$
- `head_dim` $\in \{64, 128\}$
- `num_layers` $\in \{2, 4, 8\}$
- `ffn_mult` $\in \{4, 5.25\}$ with `intermediate_size` rounded to a
  multiple of 64
- `vocab_size` $\in \{1024, 4096\}$ for the main frame; optional extra
  subjects at `131072` if they fit (embedding-dominated VRAM)

Subjects are drawn uniformly from this **finite valid set**
Independent continuous uniforms on integer ranges are not used.

### 7.3 Sequence length and Batch size

Sequence length and batch size are not sampled independently from wide
uniforms such as `seq_len ~ U(1, 512)` and `batch_size ~ U(1, 64)`. Those tails produce out-of-memory failures on a T4 or L4 and would silently truncate the sampling frame.

Each subject therefore receives a pair \((B, S)\) from a discrete catalog,
or uniformly from the constrained set
\(\{ (B,S) : B \ge 1, S \ge 32, B \cdot S \le T_{\max} \}\), where
\(T_{\max}\) is locked from the pilot together with a second cap on
estimated peak memory versus device memory. The minimum `seq_len` of 32
matches the windowed RoPE context of Section 1,
`max(sequence_length, 32)`. The same \((B, S)\) is reused for both
precisions, so that precision remains a within-subject factor as in
Section 4.

### 7.4 Assignment

For each accepted subject, measurements proceed as a within-subject
factorial of phase and precision:

1. Zepto is evaluated once per precision on the shared `(options, B, S)`.
   These estimates are analytical and do not consume a scored CUDA window.
2. The HuggingFace / PyTorch twin is built at that precision. Inference is
   measured first; training then uses one unscored warmup step and **K = 1**
   scored step, as in Section 4.
3. The twin is torn down
4. The procedure is repeated at the other precision on the **same**
   `(options, B, S)`.

Precision is therefore not assigned by a coin flip across subjects.
Duplicate `configuration_id`s are rejected so that the \(n\) of Section 6
counts unique subjects.

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

@manual{gpower2020mac,
  title        = {{G*Power} (Version 3.1.9.6) [Computer software]},
  author       = {Faul, Franz and Erdfelder, Edgar and Lang, Albert-Georg
                  and Buchner, Axel},
  organization = {Heinrich-Heine-Universit{\"a}t D{\"u}sseldorf},
  address      = {D{\"u}sseldorf, Germany},
  year         = {2020},
  url          = {https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower}
}

@manual{toster,
  title  = {{TOSTER}: Two One-Sided Tests ({TOST}) Equivalence Testing},
  author = {Lakens, Dani{\"e}l and Caldwell, Aaron},
  year   = {2024},
  url    = {https://CRAN.R-project.org/package=TOSTER},
  note   = {R package; use \texttt{power\_t\_TOST} for one-sample TOST sample size}
}
```
