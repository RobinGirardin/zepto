# Module Building Plan

Living plan for implementing transformer **Modules** in Zepto — the PyTorch-like
composition layer that sits above the operation contract and feeds module-aware
lowering. This document tracks four exploration steps:

1. **Inventory** — which layers to implement or update, scoped to Atto parity and
   the Apertus architecture
2. **Fusion policy** — for each Module, whether a fused-kernel `RegionImplementation`
   is required for correct cost estimation
3. **Pilot module** — first real Module to validate the user building experience
4. **Rollout** — phased implementation order and fused-kernel specifications

Update this file as each step is explored. Fused-kernel math follows the style of
[docs/primitive-backward.md](docs/primitive-backward.md). Lowering mechanics are
documented in [module-lowering-plan.md](module-lowering-plan.md). Primitive gaps
are tracked in [operation-to-add.md](operation-to-add.md).

---

## Current baseline (2026-08-24)

### Module composition API — complete

| Capability | Status |
|------------|--------|
| `Module`, `GraphCompositionContext`, `build_graph()` | ✅ |
| `GraphParameter`, parameter ports, `Linear` + `LinearMatMul` | ✅ |
| `module_kind`, registered submodule names in provenance | ✅ |
| Region lowering (ReLU, Linear, LayerNorm pattern) | ✅ |
| `tests/test_parameters.py`, `tests/test_region_lowering.py` | ✅ |

### Shipped primitives (enough for first real modules)

`Add`, `Subtract`, `Multiply`, `Divide`, `MatMul`, `LinearMatMul`, `Maximum`,
`Minimum`, `SquareRoot`, `Identity`, `Reshape`, `Transpose`, `Split`, `ReduceSum`,
`Pow`, `Exp`, `Log`, `Sin`, `Cos`, `Concat`, `Gather`, `RepeatKV`, `Where`,
`MaterializedCausalMask`

### Shipped modules

| Module | Structural graph | Fused lowering |
|--------|------------------|----------------|
| `Linear` | ✅ `linear_matmul` + weight param | ✅ `region/linear` |
| `LayerNorm` | ❌ placeholder (`identity`) | ✅ `region/layernorm` (awaiting real decomposition) |
| `RMSNorm` | ❌ placeholder (`identity`) | ❌ not registered |
| `RoPE` | ❌ placeholder (`identity`) | ❌ |

### Atto reference surface (target parity)

Atto recipes under `../atto/src/atto/recipes/` define the layer inventory Atto
already estimates. **Apertus** (`apertus.py`) is the primary target architecture:

```text
Apertus
├── Embedding
├── MaterializedCausalMask (shared, model-level)
├── RoPE (materialize cache + apply strategy)
├── ApertusDecoderBlock × L
│   ├── RMSNorm (pre-attention)
│   ├── GroupedQueryAttention + QKNormRMSNormStrategy + RoPE strategy
│   ├── ResidualAdd (attention)
│   ├── RMSNorm (pre-FFN)
│   ├── FFN (XIELU activation)
│   └── ResidualAdd (FFN)
├── RMSNorm (final)
└── LMHead (untied)
```

Atto also ships **VaswaniDecoder** recipes (LayerNorm, MHA, ReLU FFN,
SinusoidalPE, CrossEntropy) — lower priority unless Vaswani parity is requested.

---

## Step 1 — Layer inventory

Layers are Zepto **Modules**: provenance boundaries that compose primitives. They
are not structural graph nodes unless matched by a fused region at lowering time.

### Priority A — required for Apertus end-to-end

| # | Module | Atto counterpart | Zepto status | Notes |
|---|--------|------------------|--------------|-------|
| A1 | `RMSNorm` | `atto.recipes.norm.RMSNorm` | Placeholder | Pre-norm ×2 per block + final norm; Apertus uses **γ only** (no β) |
| A2 | `Softmax` | `atto.ops.elementwise.Softmax` | Missing | Decomposed: `Exp → ReduceSum → Divide`; scale `1/√d_h` folded in GQA |
| A3 | `RoPEMaterialize` | `atto.ops.rope.RoPEMaterialize` | Missing | Persistent `cos`/`sin` cache; Llama3 scaling is init-only for Apertus |
| A4 | `RoPEApply` | `atto.ops.rope.RoPE` via `RoPEStrategy` | Placeholder (`RoPE`) | Applies rotation on headed Q/K after split |
| A5 | `Embedding` | `atto.recipes.embedding.Embedding` | Missing | `Gather` on `(d, V)` weight; token ids as graph input metadata |
| A6 | `LMHead` | `atto.recipes.embedding.LMHead` | Missing | Untied: `Linear(d → V)` on final hidden states |
| A7 | `XIELU` | `atto.ops.elementwise.XIELU` | Missing | Apertus FFN activation; trainable `α_p`, `α_n` scalars |
| A8 | `FFN` | `atto.recipes.ffn.FFN` | Missing | `Linear(d→d_ff) → XIELU → Linear(d_ff→d)`; bias-free |
| A9 | `GroupedQueryAttention` | `atto.recipes.gqa.GroupedQueryAttention` | Missing | Q/K/V proj, QK-Norm, RoPE, RepeatKV, masked softmax, output proj |
| A10 | `ApertusDecoderBlock` | `atto.recipes.apertus.ApertusDecoderBlock` | Missing | Container: pre-RMSNorm residuals |
| A11 | `Apertus` | `atto.recipes.apertus.Apertus` | Missing | Full stack: embed, shared mask, shared RoPE, L blocks, final norm, lm_head |
| A12 | `MaterializedCausalMask` | `atto.ops.materialized_mask.MaterializedCausalMask` | Op exists | Model-level shared buffer `(1, S, S)`; may stay as functional helper |

### Priority B — Atto parity / Vaswani decoder

| # | Module | Atto counterpart | Needed for Apertus? |
|---|--------|------------------|---------------------|
| B1 | `LayerNorm` | `atto.recipes.norm.LayerNorm` | No (Apertus uses RMSNorm) |
| B2 | `MultiHeadAttention` | `atto.recipes.mha.MultiHeadAttention` | No (GQA supersedes for Apertus) |
| B3 | `DecoderBlock` | `atto.recipes.decoder_block.DecoderBlock` | No |
| B4 | `VaswaniDecoder` | `atto.recipes.vaswani_decoder.VaswaniDecoder` | No |
| B5 | `SinusoidalPE` | `atto.recipes.pe.SinusoidalPE` | No |
| B6 | `CrossEntropy` | (loss recipe) | Train-only; defer until training estimates |

### Priority C — ergonomics / containers

| # | Module | Purpose |
|---|--------|---------|
| C1 | `Sequential` | Ordered submodule list (optional; explicit `register_module` works today) |
| C2 | `ResidualAdd` | Thin wrapper over `add` for readable block recipes (optional) |
| C3 | `QKNormRMSNorm` | Strategy object or nested module inside GQA (mirror Atto `QKNormRMSNormStrategy`) |

### Updates to existing modules

| Module | Update |
|--------|--------|
| `RMSNorm` | Replace `identity` with primitive composition; support `affine=False` (γ only) for Apertus |
| `LayerNorm` | Replace `identity` with six-op chain matching `region/layernorm` pattern |
| `RoPE` | Split into `RoPEMaterialize` (cache) + `RoPEApply` (in-attention); rename or deprecate monolithic stub |
| `Linear` | Optional bias parameter (Apertus is bias-free; Vaswani may need it later) |

---

## Step 2 — Fusion policy

**Rule of thumb:** register a fused `RegionImplementation` when Atto treats the
construct as a **cost leaf** (closed-form FLOPs, elided intermediate VRAM) or when
eager decomposition would **mis-estimate** peak memory (attention, masked softmax).

| Module | Fused kernel? | Priority | Rationale |
|--------|---------------|----------|-----------|
| `Linear` | ✅ Yes (`region/linear`) | Done | Provenance envelope around one `linear_matmul` |
| `LayerNorm` | ✅ Yes (`region/layernorm`) | Done (impl); module pending | Atto `LayerNorm` op ≈ `7·S·d` FLOP leaf; fusion elides mean/var temps |
| `RMSNorm` | ✅ Yes (`region/rmsnorm`) | **High** | Atto `RMSNorm` op ≈ `5·S·d` (γ only: `4·S·d`); same elision argument |
| `Softmax` | ✅ Yes (`region/softmax`) | **High** | Attention bottleneck; masked variant needed for Apertus |
| `Softmax` + causal mask | ✅ Yes (`region/masked_softmax`) | Medium | Fuses scale + mask add + softmax for score tensor |
| `ReLU` | ✅ Yes (`region/relu`) | Done | Pattern on `maximum(x, 0)` |
| `XIELU` | ⚠️ Optional (`region/xielu`) | Medium | Decomposed path works; fusion matches Atto leaf + `cuda_kernel` transient policy |
| `RoPEApply` | ⚠️ Optional later | Low | Atto documents decomposed sin/cos/mul/add FLOPs; fusion saves temp VRAM |
| `RoPEMaterialize` | ❌ No | — | Persistent cache; billed once, not a per-forward fusion |
| `Embedding` | ❌ No | — | Single `Gather`; op contract is sufficient |
| `FFN` | ❌ No (aggregated) | — | Sum of Linear + XIELU + Linear per-op/region costs |
| `GroupedQueryAttention` | ⚠️ Optional later (`region/gqa` / flash) | Low | Default: per-op route with provenance rollup; Flash-style fusion is Phase 2 |
| `MultiHeadAttention` | ⚠️ Optional later | Low | Same as GQA without RepeatKV |
| `MaterializedCausalMask` | ❌ No | — | One-shot buffer allocation |
| `ApertusDecoderBlock`, `Apertus` | ❌ No | — | Containers only |
| `CrossEntropy` | ⚠️ Optional later | Defer | Loss region fusion |

### Cost authority reminder

- **Unfused (aggregated) modules:** total cost = sum of per-op lowered nodes;
  provenance attributes FLOPs/VRAM to module paths in Phase D.
- **Fused modules:** `RegionImplementation` declares closed-form FLOPs and
  explicit `ResourceEvent`s; must not double-count internal primitives.

---

## Step 3 — Pilot module: `RMSNorm`

**Goal:** validate that a module author can (1) declare parameters in `__init__`,
(2) compose primitives in `forward()`, (3) build a graph with `build_graph()`,
(4) lower with optional fusion, and (5) match Atto golden costs.

**Why RMSNorm first (not LayerNorm or GQA)?**

| Candidate | Pros | Cons |
|-----------|------|------|
| `Linear` | Already done | Does not exercise new composition patterns |
| **`RMSNorm`** | **Apertus-critical; simpler than LayerNorm; Atto golden tests exist; needs new fused region** | Requires careful reduce-axis + broadcast wiring |
| `Softmax` | Needed for attention | Three-op chain + scale; better as second pilot |
| `Embedding` | Simple | No backward complexity; less lowering coverage |

### Pilot acceptance criteria

- [ ] `RMSNorm(normalized_shape=d, eps=ε, affine=True|False)` builds a valid structural graph
- [ ] Rank-2 input `(S, d)`; normalization axis = last dim (configurable later)
- [ ] Parameters: `weight` (γ), optional `bias` (β) when `affine=True`
- [ ] Provenance: all ops tagged `component_type="RMSNorm"` under registered name
- [ ] Unfused lowering: N lowered ops (identity or per-op route)
- [ ] Fused lowering: one `region/rmsnorm` node with Atto-matching FLOPs
- [ ] Test: `tests/test_modules.py::test_rmsnorm_graph_and_lowering`
- [ ] Golden parity: compare forward/backward FLOPs against `../atto/tests/analyze/rmsnorm/`

### Pilot composition sketch (Apertus: γ only, `affine=False`)

For input `x` of shape `(S, d)` and learnable γ of shape `(d,)`:

```text
sq      = x * x                          # elementwise, (S, d)
ms      = reduce_sum(sq, axis=-1, keepdim=True) / d   # (S, 1) — or fold /d into later step
rstd    = rsqrt(ms + eps)                # (S, 1) via add + sqrt + divide, or dedicated rsqrt path
normed  = x * rstd                       # broadcast multiply, (S, d)
out     = normed * γ                     # broadcast multiply, (S, d)
```

Exact op sequence TBD during implementation; must align with `region/rmsnorm`
pattern rule once registered.

### Pilot UX checklist (qualitative)

- [ ] Parameter creation inside `__init__` reads naturally (mirror `Linear`)
- [ ] No manual `GraphCompositionContext.current()` except in tests / `build_graph` factory
- [ ] Nested use: `self.pre_attn_norm = RMSNorm(d); self.register_module("pre_attn_norm", …)`
- [ ] Error messages for rank/shape mismatch are actionable

---

## Step 4 — Implementation roadmap

### Phase M0 — Pilot + fused RMSNorm

1. Implement `RMSNorm.forward()` primitive composition (`src/zepto/modules/rms_norm.py`)
2. Add `tests/test_modules.py` with graph shape, parameter binding, provenance tests
3. Register `region/rmsnorm` in `src/zepto/core/lowering/implementations/regions/`
4. Document fused kernel spec (§ Fused kernel specs below)
5. Atto golden smoke (manual or scripted compare)

### Phase M1 — Norm + activation leaves

1. Implement real `LayerNorm` decomposition (unlock existing `region/layernorm`)
2. Implement `Softmax` Module + `region/softmax`
3. Implement `XIELU` Module (decomposed); optional `region/xielu`

### Phase M2 — Positional encoding + embeddings

1. Split `RoPE` into `RoPEMaterialize` + `RoPEApply`
2. Implement `Embedding`, `LMHead`
3. Wire `MaterializedCausalMask` at model level (functional or thin Module)

### Phase M3 — Attention

1. Implement `Softmax` scale + `MaterializedCausalMask` add path inside scores
2. Implement `QKNormRMSNorm` helper (two nested `RMSNorm` on headed Q/K)
3. Implement `GroupedQueryAttention` (decomposed, per-op lowering)
4. Optional: `region/masked_softmax`, later `region/gqa`

### Phase M4 — Blocks + Apertus

1. `FFN` Module
2. `ApertusDecoderBlock` container
3. `Apertus` root Module with presets (`apertus_8b`, `apertus_70b` factories)
4. End-to-end graph build test matching Atto `Apertus.build_detailed()` topology

### Phase M5 — Vaswani parity (optional)

1. `MultiHeadAttention`, `DecoderBlock`, `VaswaniDecoder`
2. `LayerNorm` in Vaswani blocks (already have module stub)
3. `CrossEntropy` for training estimates

---

## Fused kernel specifications

Each fused region gets a subsection here (and eventually a dedicated doc under
`docs/fused/` or an appendix in `docs/primitive-backward.md`). Format mirrors
primitive backward docs: forward math, backward math, broadcast rules, memory.

Spec status: ⬜ draft · 🟡 in review · ✅ shipped

---

### `region/rmsnorm` — RMSNorm ⬜

**Matches Module:** `RMSNorm`  
**Atto reference:** `atto.ops.norm.RMSNorm`, framework leaf `41 - rmsnorm.md`

#### Forward

For each token vector $x \in \mathbb{R}^d$ (row of input $X \in \mathbb{R}^{S \times d}$):

$$
\text{ms}(x) = \frac{1}{d}\sum_{i=1}^{d} x_i^2, \qquad
r = \sqrt{\text{ms}(x) + \epsilon}
$$

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{r} \quad (\text{Apertus: no } \beta)
$$

With affine shift (full form):

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{r} + \beta
$$

**Forward FLOPs (per token, γ only):** $\approx 4d$ (square, mean-of-squares, rsqrt, normalize, scale)  
**Forward FLOPs (full affine):** $\approx 5d$  
**Batch forward:** multiply by $S$; block uses $2L$ norms → $10 \cdot S \cdot d \cdot L$ (γ only)

#### Backward

Let $y = x / r$, $z = \gamma \odot y$. Upstream gradient $\bar{z} = \partial L / \partial z$.

$$
\bar{\gamma} = \sum_{\text{tokens}} \bar{z} \odot y
$$

$$
\bar{x} = \frac{\gamma \odot \bar{z}}{r} - \frac{x}{d \cdot r^3} \sum_i (\gamma_i \bar{z}_i x_i)
$$

(last term is the RMS statistic backward; implementation may save $r$ or `rstd` $= 1/r$.)

**Backward FLOPs:** Atto convention ≈ $2 \times$ forward for train.

#### Broadcast

- γ (and β): shape $(d,)$ broadcast over token dimension $(S, d)$
- `ms`, `r`: shape $(S, 1)$ broadcast over $(S, d)$
- Gradient-shape invariant: $\bar{x}$ same shape as $x$; $\bar{\gamma}$ reduced to $(d,)$

#### Memory (unfused decomposition — structural default)

| Tensor | Role | When saved |
|--------|------|------------|
| $x$ | input | Backward if `requires_grad` |
| `ms` or `rstd` | auxiliary | Forward SAVE → backward |
| $y$ | optional temp | Recompute from $x$, $r$ preferred |

#### Memory (fused `region/rmsnorm`)

| Tensor | Role | Policy |
|--------|------|--------|
| output | activation | ALLOCATE → PERSIST |
| `rstd` | saved stat | ALLOCATE → SAVE (per Atto: $S$ scalars × 2 norms × $L$) |
| internal `sq`, `ms` | — | **Elided** (not allocated in fused lowering) |

**Declared forward FLOPs:** `5 * numel(x)` (full affine) or `4 * numel(x)` (γ only) — tune to Atto golden.  
**Declared backward FLOPs:** `2 * forward` when any input requires grad.

---

### `region/layernorm` — LayerNorm 🟡

**Matches Module:** `LayerNorm`  
**Status:** Lowering impl exists; Module decomposition pending.

#### Forward

Per token $x \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_i x_i, \quad
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}}, \quad
y = \gamma \odot \hat{x} + \beta
$$

**Forward FLOPs:** $\approx 7 \cdot S \cdot d$ (Atto leaf)

#### Backward

Save $x$ (Atto); optionally $\mu$, $\sigma$. Standard LayerNorm VJP; $\bar{\gamma}$, $\bar{\beta}$ reduced over tokens.

#### Broadcast / memory

Same broadcast rules as RMSNorm for γ, β. Fused region elides centered/variance temporaries; saves mean + inv_std (see existing `FusedLayerNormRegionImplementation`).

---

### `region/softmax` — Softmax ⬜

**Matches Module:** `Softmax`  
**Atto reference:** `atto.ops.elementwise.Softmax`

#### Forward

For vector $s \in \mathbb{R}^n$ (one row of scores, possibly already scaled):

$$
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

Decomposed structural path: `exp(s) → reduce_sum → divide`.

**Forward FLOPs:** $\approx 3n$ per softmax (exp, sum, div) + optional scale multiply

#### Backward

With output $p$ saved:

$$
\bar{s}_i = p_i \left(\bar{p}_i - \sum_j \bar{p}_j p_j\right)
$$

**Backward FLOPs:** $\approx 2 \times$ forward (Atto convention)

#### Broadcast

Softmax axis is typically last dim of score tensor $(h, S, S)$ or $(S, S)$;
reduction axis = softmax dim. Upstream $\bar{p}$ same shape as $p$.

#### Memory

| Mode | Saved |
|------|-------|
| Unfused | $p$ (output) PERSIST; exp temp ALLOCATE→RELEASE |
| Fused | $p$ SAVE; exp temp elided |

---

### `region/masked_softmax` — scaled causal softmax ⬜

**Matches region inside:** `GroupedQueryAttention` score step  
**Atto reference:** GQA recipe + `MaterializedCausalMask` add before Softmax

#### Forward

$$
s' = \frac{QK^\top}{\sqrt{d_h}} + M, \qquad P = \text{softmax}(s')
$$

$M$ is materialized additive mask $(1, S, S)$ broadcast to $(h, S, S)$.

#### Fusion value

Single region replaces `divide → add → exp → reduce_sum → divide` chain;
declares Atto GQA score+softmax FLOP bucket without materializing pre-softmax
$s'$ if policy allows (configurable: eager saves $s'$, fused may save only $P$).

#### Memory

| Tensor | Apertus eager | Fused option |
|--------|---------------|--------------|
| Mask $M$ | Shared persistent buffer | Same (model-level) |
| Scores $s'$ | Activation | Elide if only $P$ needed for backward |
| Weights $P$ | Activation SAVE | SAVE |

*Detailed backward for masked softmax: TBD when GQA Module lands.*

---

### `region/xielu` — xIELU ⬜ (optional)

**Atto reference:** `55 - xielu.md`, `XIELU(style="cuda_kernel"|"python")`

Piecewise activation with trainable scalars $\alpha_p$, $\alpha_n$. Decomposed
via `Where`, `Exp`, `Pow`, `Multiply`, `Add`. Fusion matches Atto leaf FLOPs and
`cuda_kernel` vs `python` transient policy ($|H|$ buffer for python path only).

*Full spec deferred to Phase M1.*

---

## Atto ↔ Zepto mapping (quick reference)

| Atto recipe / op | Zepto Module | Primary primitives |
|------------------|--------------|-------------------|
| `RMSNorm` | `RMSNorm` | ReduceSum, Mul, Add, Sqrt, Div + γ |
| `LayerNorm` | `LayerNorm` | ReduceSum, Sub, Mul, Add, Sqrt, Div + γ, β |
| `Softmax` | `Softmax` | Exp, ReduceSum, Div |
| `XIELU` | `XIELU` | Where, Exp, Pow, Mul, Add + α params |
| `EmbeddingLookup` | `Embedding` | Gather |
| `RoPEMaterialize` | `RoPEMaterialize` | Sin, Cos, Gather/MatMul on inv_freq |
| `RoPE` | `RoPEApply` | Split, Mul, Add on half-dims |
| `GroupedQueryAttention` | `GroupedQueryAttention` | Linear×4, Reshape, Transpose, RepeatKV, Add, Softmax, MatMul |
| `FFN` | `FFN` | Linear, XIELU, Linear |
| `MaterializedCausalMask` | (functional) | `materialized_causal_mask` op |
| `ApertusDecoderBlock` | `ApertusDecoderBlock` | composes above |
| `Apertus` | `Apertus` | composes above |

---

## Open questions (explore & update)

1. **Reduce-mean vs reduce-sum:** Zepto has `ReduceSum` only. RMSNorm uses
   `sum(x²)/d` — implement as `reduce_sum / d` via constant divide, or add
   `ReduceMean` op? (*Lean toward sum + scalar divide to avoid new op.*)

2. **RoPE module split:** One `RoPE` class with `materialize()` + `apply()`, or
   two Modules matching Atto ops? (*Two modules align with Atto cost docs.*)

3. **Token IDs in Embedding:** Graph input metadata only (Atto pattern), or
   explicit `GraphTensor` for `(S,)` int indices?

4. **QK-Norm as nested Module vs strategy object:** Atto uses a strategy protocol;
   Zepto may use nested `RMSNorm(d_h)` modules for simpler provenance.

5. **Apertus Llama3 RoPE scaling:** Atto documents init-only θ/λ; confirm Zepto
   `RoPEMaterialize` exposes same knobs without billing scaling in step FLOPs.

6. **GQA phase intent:** Train vs inference KV tagging — mirror Atto `PhaseIntent`
   via `ValueMetadata.semantic_type` / role on K/V tensors.

7. **First Atto golden target:** RMSNorm only, or RMSNorm + Linear block?

---

## Checklist (living)

### Step 1 — Inventory
- [x] Map Apertus layer stack from Atto
- [x] Map Atto full recipe exports
- [ ] Confirm with user: Vaswani parity in scope?
- [ ] Confirm Apertus β-less RMSNorm as default

### Step 2 — Fusion policy
- [x] Table per Module
- [ ] Review `region/rmsnorm` priority vs `region/softmax`

### Step 3 — Pilot
- [ ] Implement `RMSNorm` composition
- [ ] Add `test_modules.py`
- [ ] UX review notes appended below

### Step 4 — Rollout
- [ ] Phase M0 complete
- [ ] Fused specs promoted to `docs/fused/` when stable

---

## Revision log

| Date | Change |
|------|--------|
| 2026-08-24 | Initial plan: inventory, fusion matrix, RMSNorm pilot, roadmap, draft fused specs |
