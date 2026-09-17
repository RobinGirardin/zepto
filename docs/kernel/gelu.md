# Zepto kernel research: GELU (Gaussian Error Linear Unit)

**Date:** 2026-09-14
**Proposer:** Robin Girardin
**Scope:** Standalone elementwise activation leaf; three approximation variants (exact `erf`, tanh, quick sigmoid); prefill + decode; training + inference; CUDA primary; GeGLU gate-branch cross-ref
**Context:** GELU is the default MLP nonlinearity in BERT-style encoders, ViT vision blocks, and GeGLU FFNs (Gemma, Muse-Glimmer, Qwen3-VL vision). Zepto modules and recipes exist (`GELUTanh`, `GELUErf`, `GeluRecipe`); fused `RegionImplementation`s are **not yet registered**. Default Zepto estimates use the **fused region leaf**, not a sum of unfused primitives.

---

## Section 0: Mathematical definition

GELU ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)) applies the standard-normal cumulative distribution function \(\Phi\) elementwise:

\[
\mathrm{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left(1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right).
\]

**Three production variants** (all elementwise, same I/O shape):

| Variant | Formula | Typical use |
|---------|---------|-------------|
| **Exact (`erf`)** | \(\tfrac{x}{2}(1 + \mathrm{erf}(x/\sqrt{2}))\) | BERT `GELUActivation`, PyTorch `F.gelu(approximate="none")` |
| **Tanh (GPT/BERT-NewGELU)** | \(\tfrac{x}{2}\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right)\) | GPT-2, LLaMA-era encoders, `NewGELUActivation`, `F.gelu(approximate="tanh")` |
| **Quick (sigmoid)** | \(x \cdot \sigma(1.702\,x)\) | CLIP / OpenAI QuickGELU ([hendrycks/GELUs](https://github.com/hendrycks/GELUs)) |

**I/O shapes:** input \(X \in \mathbb{R}^{S \times d}\) (Zepto rank-2; HF/vLLM often \((B,T,d)\)); output \(Y\) same shape. No reduction axes.

**Numerics policies:**
- Special functions (`erf`, `tanh`, `exp`/`sigmoid`) bill **4 FLOPs per element** (Zepto coarse bucket; `docs/kernel-implementation.md` §1, `src/zepto/semantic/operations/erf.py`, `tanh.py`).
- Lone `mul` / `add` = **1 FLOP** each; `pow(x,3)` decomposes to \(x^2\) then \(x^2 \cdot x\) (2 FLOPs).
- Comparisons and branch selection: **0 FLOPs** (ReLU policy).
- PyTorch `aten::gelu` dispatches fused CUDA/CPU kernels for both `GeluType::None` (erf) and `GeluType::Tanh` ([`ActivationGeluKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)).

**Structural vs eager decomposition:**
- **Erf path (identity):** `Multiply(x, inv_sqrt2) → Erf → Add(1) → Multiply(x) → Multiply(0.5)` — five ops, multiple HBM-resident temps.
- **Tanh path (identity):** nine-op chain in `src/zepto/modules/gelu.py` (`x² → x³ → κx³ → add → scale → tanh → 1+tanh → 0.5x → mul`).
- **Fused leaf:** one elementwise kernel; intermediates (\(x^2\), tanh input, \(\sigma\) input) stay in registers.

**GeGLU composition (parent module, not this leaf):**
\[
\mathrm{GeGLU}(x) = \mathrm{GELU}(W_g x) \odot (W_u x)
\]
([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). Liger fuses tanh-GELU inside `geglu` Triton kernels ([`liger_kernel/ops/geglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py)).

**Zepto identity references:** `src/zepto/modules/gelu.py` — `GELUErf`, `GELUTanh`.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch `aten::gelu` / `gelu_backward` (erf + tanh variants); HF `GELUActivation`, `NewGELUActivation`, `GELUTanh` → `F.gelu`; Liger fused GeGLU (tanh-GELU inside) |
| **Identity lowering** | Unfused semantic chain | Zepto `GELUErf` (5 ops) or `GELUTanh` (9 ops); HF Python fallbacks in [`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gelu` (tanh, **12/20** FLOPs/elem), `region/gelu_erf` (exact, **8/17**), future `region/gelu/quick` (**6/9**) — see `src/zepto/analysis/lowering/recipes/gelu.py` |

**Default estimates use the fused region leaf for the selected approximation — not the sum of identity primitive FLOPs when fusion is selected.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **PyTorch ATen (erf)** | `F.gelu(x, approximate="none")`, `aten::gelu` | Yes | standalone | CUDA / CPU / MPS / XPU | Both; autograd saves **input** |
| **PyTorch ATen (tanh)** | `F.gelu(x, approximate="tanh")` | Yes | standalone | CUDA / CPU / MPS / XPU | Both; autograd saves **input** |
| **PyTorch backward** | `gelu_backward`, `GeluBackwardCUDAKernelImpl` | Yes | standalone | CUDA primary | Recomputes tanh/erf from saved input |
| **HF `GELUActivation`** | [`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) → `F.gelu` | Yes (ATen) | standalone | Any | Both |
| **HF `NewGELUActivation` / `GELUTanh`** | Python tanh formula or `F.gelu(approximate="tanh")` | Yes | standalone | Any | Both |
| **HF `QuickGELUActivation`** | `x * sigmoid(1.702*x)` | Decomposed eager | standalone | Any | Both |
| **HF `FastGELUActivation`** | Alternate tanh coeff (0.7978845608) | Decomposed eager | standalone | Any | Both |
| **HF Hub kernels** | `@use_kernel_forward_from_hub("GeLU")`, `"NewGELU"`, `"QuickGELU"`, `"FastGELU"` | Yes (Hub) | standalone | Hub-routed when `use_kernels=True` | Both |
| **Liger GeGLU** | [`liger-kernel`](https://github.com/linkedin/Liger-Kernel) / TRL `use_liger_kernel=True` | Yes (GELU inside GeGLU) | MLP gate branch | CUDA (Triton) | Training-focused |
| **vLLM / SGLang ViT MLPs** | Model-specific GELU-tanh blocks | Partial | MLP / vision | CUDA primary | Both |
| **Zepto (identity)** | `modules/gelu.py` decomposed chains | No | standalone | Any | Both |
| **Zepto (recipe, unregistered)** | `recipes/gelu.py`, `regions/gelu/rules.py` | Yes (cost leaf) | standalone | `any` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **elementwise activation leaf** — not an attention fusion boundary (A/B/C/D). Classify as `standalone_activation` (same family as `region/silu`, `region/relu`, `region/xielu`).

**Mutually exclusive (per invocation):**
- `region/gelu` (tanh) vs `region/gelu_erf` (exact) vs `region/gelu/quick` — selected by approximation variant / pattern match.
- Each fused region vs its identity lowering chain — region wins at priority 10 when pattern matches (`pat-gelu-tanh-decomposed`, `pat-gelu-erf-decomposed`, provenance rules on `GELUTanh` / `GELUErf`).

**Composable:**
- **GeGLU FFN:** `Linear(gate) → region/gelu/* → Multiply(gate_act, up) → Linear(down)`. GELU region covers only the gate nonlinearity; gate×up multiply and three GEMMs are separate leaves.
- **Liger GeGLU** replaces post-GEMM GELU+mul with one Triton kernel — Zepto must not double-count `region/gelu` when a future `region/geglu/liger` leaf is selected.
- **ViT MLP:** `Linear → GELU → Linear` — GELU is a sequential leaf between GEMMs; no overlap with attention regions.
- Does not overlap `region/softmax`, `region/gqa/*`, `region/rmsnorm`, etc.

**Execution constraints:**
- Tanh pattern requires nine-op family sequence with `gelu_tanh_activation` constraint.
- Erf pattern requires five-op chain with `gelu_erf_activation` constraint.
- Provenance rules require contiguous in-graph order for `GELUTanh` / `GELUErf` modules.
- `requires_grad=False` → backward FLOPs = 0; no SAVE events.
- Apertus-8B uses **xIELU**, not GELU, in its FFN; GELU costing applies to BERT/ViT/Gemma/GeGLU stacks at width \(d_{\mathrm{ff}}\).

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert Y \rvert = S \cdot d\) (or \(S \cdot d_{\mathrm{ff}}\) for an MLP intermediate).

### 4.1 Identity lowering — exact erf (`GELUErf`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Multiply(x, inv_sqrt2)` | mul | 1 | \(n\) |
| 2 | `Erf(scaled)` | special | 4 | \(4n\) |
| 3 | `Add(1, erf_out)` | add | 1 | \(n\) |
| 4 | `Multiply(x, one_plus_erf)` | mul | 1 | \(n\) |
| 5 | `Multiply(0.5, x_times_cdf)` | mul | 1 | \(n\) |
| **Identity total (erf)** | | | | **\(8n\)** |

Unfused forward allocates up to **4** full-rank temps (`scaled`, `erf_out`, `one_plus_erf`, `x_times_cdf`) plus \(Y\).

### 4.2 Identity lowering — tanh (`GELUTanh`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Multiply(x, x)` → \(x^2\) | mul | 1 | \(n\) |
| 2 | `Multiply(x², x)` → \(x^3\) | mul | 1 | \(n\) |
| 3 | `Multiply(κ, x³)` | mul | 1 | \(n\) |
| 4 | `Add(x, κx³)` | add | 1 | \(n\) |
| 5 | `Multiply(√(2/π), inner)` | mul | 1 | \(n\) |
| 6 | `Tanh(tanh_input)` | special | 4 | \(4n\) |
| 7 | `Add(1, tanh_out)` | add | 1 | \(n\) |
| 8 | `Multiply(0.5, x)` | mul | 1 | \(n\) |
| 9 | `Multiply(half_x, 1+tanh)` | mul | 1 | \(n\) |
| **Identity total (tanh)** | | | | **\(12n\)** |

Unfused forward allocates up to **7** full-rank temps plus \(Y\).

### 4.3 Identity lowering — quick sigmoid

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Multiply(1.702, x)` | mul | 1 | \(n\) |
| 2 | `Sigmoid(1.702x)` | special | 4 | \(4n\) |
| 3 | `Multiply(x, σ(...))` | mul | 1 | \(n\) |
| **Identity total (quick)** | | | | **\(6n\)** |

### 4.4 Fused region leaf (kernel-accurate)

Fused kernels perform the same arithmetic as identity chains but elide HBM-resident intermediates. Per-element FLOP totals match identity for each variant:

| Variant | Fused leaf FLOPs/element | Region id |
|---------|--------------------------|-----------|
| Exact erf | **8** | `region/gelu_erf` |
| Tanh (GPT/NewGELU) | **12** | `region/gelu` |
| Quick sigmoid | **6** | `region/gelu/quick` (future) |

### 4.5 Paper-comparable (if different)

Hendrycks & Gimpel (2016) define GELU via \(\Phi(x)\) without separate FLOP accounting. Appendix E for Apertus **omits** activations (MLP = GEMM-only); that omission is not a lower GELU constant.

Huang & Schlag §3.5 note xIELU is **on par with GELU/SiLU** in op mix (~one special function + several mul/add). Tanh-GELU (12 fwd) is **heavier** than SiLU (5 fwd) because of the cubic term and tanh; exact erf-GELU (8 fwd) sits between SiLU and tanh-GELU.

**Closed form (kernel-accurate, per variant):**
\[
\mathrm{FLOPs}_{\mathrm{GELU,fwd}} =
\begin{cases}
8 \cdot n & \text{exact erf} \\
12 \cdot n & \text{tanh approximation} \\
6 \cdot n & \text{quick sigmoid}
\end{cases}
\]

**Arithmetic intensity (numeric example):** Gemma-style GeGLU gate, \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\), bf16 (\(e{=}2\)), tanh variant:

| Quantity | Value |
|----------|-------|
| \(n = S \cdot d_{\mathrm{ff}}\) | \(\approx 1.76 \times 10^8\) |
| FLOPs (tanh) | \(12n \approx 2.11 \times 10^9\) |
| Min HBM bytes (read \(X\) + write \(Y\), fused) | \(2 n e \approx 704\) MiB |
| Arithmetic intensity | \(\approx 0.30\) FLOP/byte — **memory-bound** |

Tanh-GELU forward at this tile is ~**3.7%** of Up+Down GEMM FLOPs (\(\approx 57\)B/layer at \(S{=}8192\)); exact erf is ~**2.5%** (\(8n\)).

---

## Section 5: Backward FLOPs — step-by-step derivation

PyTorch autograd for `aten::gelu` saves **input** \(X\) and recomputes special functions in the backward kernel ([`GeluBackwardCUDAKernelImpl`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)).

### 5.1 Exact erf — fused region leaf

Derivative: \(\mathrm{GELU}'(x) = \Phi(x) + x\,\phi(x)\) where \(\phi\) is the standard-normal PDF.

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(x / \sqrt{2}\) | mul | 1 | \(n\) |
| 2 | Recompute \(\mathrm{erf}(x/\sqrt{2})\) | special | 4 | \(4n\) |
| 3 | \(\Phi = \tfrac{1}{2}(1 + \mathrm{erf})\) | add + mul | 2 | \(2n\) |
| 4 | \(x^2\) for PDF | mul | 1 | \(n\) |
| 5 | \(\exp(-x^2/2)\) | special | 4 | \(4n\) |
| 6 | PDF scale (\(\phi\) constants) | mul + mul | 2 | \(2n\) |
| 7 | \(x \cdot \phi(x)\) | mul | 1 | \(n\) |
| 8 | \(\Phi + x\phi\) | add | 1 | \(n\) |
| 9 | `grad_out * factor` | mul | 1 | \(n\) |
| **Fused backward (erf)** | | | | **\(17n\)** |

### 5.2 Tanh — fused region leaf

Backward follows chain rule through \( \tfrac{x}{2}(1+\tanh(u)) \) with \(u = \sqrt{2/\pi}(x + 0.044715 x^3)\) ([CUDA kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)).

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1–5 | Recompute \(x^2, x^3, u\) (same as forward steps 1–5) | mul/add | 5 | \(5n\) |
| 6 | Recompute \(\tanh(u)\) | special | 4 | \(4n\) |
| 7 | \(\tanh^2\), \(1-\tanh^2\) | mul + sub | 2 | \(2n\) |
| 8 | Inner deriv \(1 + 3 \cdot 0.044715 \cdot x^2\) | mul + mul + add | 3 | \(3n\) |
| 9 | Scale inner deriv by \(\sqrt{2/\pi}\) | mul | 1 | \(n\) |
| 10 | \(\tfrac{x}{2}(1+\tanh)\) branch + tanh-chain branch | mul + mul + add | 3 | \(3n\) |
| 11 | `grad_out * combined_deriv` | mul | 1 | \(n\) |
| **Fused backward (tanh)** | | | | **\(20n\)** |

### 5.3 Quick sigmoid — fused region leaf

\(y = x \cdot \sigma(1.702 x)\); derivative matches scaled-SiLU: \(\sigma(1.702x) + 1.702\,x\,\sigma(1.702x)(1-\sigma(1.702x))\).

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(1.702 \cdot x\) | mul | 1 | \(n\) |
| 2 | Recompute \(\sigma(1.702x)\) | special | 4 | \(4n\) |
| 3 | \((1-\sigma)\), \(x(1-\sigma)\), factor assembly | sub + mul + add + mul | 4 | \(4n\) |
| 4 | `grad_out * factor` | mul | 1 | \(n\) |
| **Fused backward (quick)** | | | | **\(9n\)** |

`requires_grad=False` → backward FLOPs = **0** for all variants.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GELU,bwd}} =
\begin{cases}
17 \cdot n & \text{exact erf, if requires\_grad} \\
20 \cdot n & \text{tanh, if requires\_grad} \\
9 \cdot n & \text{quick, if requires\_grad} \\
0 & \text{otherwise}
\end{cases}
\]

**Common mistake:** Do **not** use \(2 \times \mathrm{fwd}\) (GEMM heuristic) or add forward + backward special-function counts (double-counts `exp`/`erf`/`tanh` inside the VJP). Matches `GeluRecipe` in `src/zepto/analysis/lowering/recipes/gelu.py`.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| **Identity erf (unfused)** | `scaled`, `erf_out`, `one_plus_erf`, `x_times_cdf`, \(Y\) | \(5 n e\) | — |
| **Identity tanh (unfused)** | \(x^2\), \(x^3\), `kappa_x³`, `inner`, `tanh_in`, `tanh_out`, `1+tanh`, `half_x`, \(Y\) | \(9 n e\) | — |
| **Identity quick (unfused)** | `scaled`, `sigmoid_out`, \(Y\) | \(3 n e\) | — |
| **Fused `region/gelu*`** | \(Y\) only | \(n e\) | all intermediates above |

Input \(X\) is a boundary edge (not re-allocated by the region).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| PyTorch `aten::gelu` autograd | input \(X\) | \((S,d)\) | \(n e\) | `save_for_backward(input)` |
| **`region/gelu*` (all variants)** | input \(X\) | \((S,d)\) | \(n e\) | `save_input: true` when `requires_grad` |
| Unfused identity chains | each op may SAVE input separately | — | up to \(2 n e\) redundant | superseded by fused region |

Fused GELU does **not** materialize \(x^2\), tanh/erf intermediates, or \(\sigma\) output to HBM; backward recomputes from saved \(X\) (same strategy as PyTorch fused `gelu_backward`).

### 6.3 Resource event chains

**Identity lowering — tanh (unfused, abbreviated):**
```
ALLOCATE(x²) → ALLOCATE(x³) → ALLOCATE(κx³) → ALLOCATE(inner) → ALLOCATE(tanh_in)
  → ALLOCATE(tanh_out) → ALLOCATE(1+tanh) → ALLOCATE(half_x) → ALLOCATE(Y)
  → SAVE(input) on each op requiring grad
```

**Identity lowering — erf (unfused):**
```
ALLOCATE(scaled) → ALLOCATE(erf_out) → ALLOCATE(1+erf) → ALLOCATE(x·cdf) → ALLOCATE(Y)
  → SAVE(input)
```

**Fused region leaf (`region/gelu`, `region/gelu_erf`, `region/gelu/quick`):**
```
ALLOCATE(Y) → SAVE(input) when requires_grad
```

Backward phase end: `RELEASE(input)` on the saved input edge.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| PyTorch ATen (erf) | torch | Yes | standalone | \(8n\) | \(17n\) | SAVE(input) | — |
| PyTorch ATen (tanh) | torch | Yes | standalone | \(12n\) | \(20n\) | SAVE(input) | — |
| HF `GELUActivation` | transformers | Yes (ATen) | standalone | \(8n\) | \(17n\) | SAVE(input) | — |
| HF `NewGELUActivation` | transformers | Yes | standalone | \(12n\) | \(20n\) | SAVE(input) | — |
| HF `QuickGELUActivation` | transformers | No (eager) | standalone | \(6n\) | \(9n\) | SAVE(input) | future `region/gelu/quick` |
| Liger GeGLU | liger-kernel | Yes (GELU in GeGLU) | MLP gate | \(12n\) gate slice* | \(20n\) gate slice* | Elide gate/up temps | future `region/geglu/liger` |
| Zepto identity (tanh) | 9-op chain | No | standalone | \(12n\) | higher† | Multiple SAVEs | — |
| Zepto fused (tanh) | `region/gelu` | Yes | standalone | \(12n\) | \(20n\) | SAVE(input) | **to_implement** |
| Zepto fused (erf) | `region/gelu_erf` | Yes | standalone | \(8n\) | \(17n\) | SAVE(input) | **to_implement** |

\*Gate-branch slice only; full GeGLU adds two GEMMs and one elementwise mul.  
†Identity backward sums per-op VJP billing; fused leaf is authoritative.

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gelu
recommended_region_ids:
  - id: region/gelu
    variant: tanh
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    approximate: tanh
    pattern_rule:
      op_families:
        - multiply
        - multiply
        - multiply
        - add
        - multiply
        - tanh
        - add
        - multiply
        - multiply
      constraints:
        - gelu_tanh_activation
    recipe:
      forward_flops: "12 * numel(output)"
      backward_flops: "20 * numel(output) if requires_grad else 0"
      forward_flops_per_element: 12
      backward_flops_per_element: 20
      save_input: true
      approximate: tanh
      elided_temps:
        - x_squared
        - x_cubed
        - kappa_x_cubed
        - inner_sum
        - tanh_input
        - tanh_output
        - one_plus_tanh
        - half_x
      saved_backward:
        - name: input
          shape: "(S, d)"
      resource_events_forward:
        - "ALLOCATE(Y) → SAVE(input) when requires_grad"
      resource_events_backward:
        - "RELEASE(input) on backward phase end"
      numerics_tags:
        - tanh_special_function_4_flops
        - comparison_not_billed
  - id: region/gelu_erf
    variant: erf
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    approximate: none
    pattern_rule:
      op_families:
        - multiply
        - erf
        - add
        - multiply
        - multiply
      constraints:
        - gelu_erf_activation
    recipe:
      forward_flops: "8 * numel(output)"
      backward_flops: "17 * numel(output) if requires_grad else 0"
      forward_flops_per_element: 8
      backward_flops_per_element: 17
      save_input: true
      approximate: none
      elided_temps:
        - scaled
        - erf_output
        - one_plus_erf
        - x_times_cdf
      saved_backward:
        - name: input
          shape: "(S, d)"
      resource_events_forward:
        - "ALLOCATE(Y) → SAVE(input) when requires_grad"
      resource_events_backward:
        - "RELEASE(input) on backward phase end"
      numerics_tags:
        - erf_special_function_4_flops
  - id: region/gelu/quick
    variant: quick
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    approximate: quick
    pattern_rule:
      op_families:
        - multiply
        - sigmoid
        - multiply
    recipe:
      forward_flops: "6 * numel(output)"
      backward_flops: "9 * numel(output) if requires_grad else 0"
      forward_flops_per_element: 6
      backward_flops_per_element: 9
      save_input: true
      approximate: quick
      elided_temps:
        - scaled_input
        - sigmoid_output
      saved_backward:
        - name: input
          shape: "(S, d)"
      resource_events_forward:
        - "ALLOCATE(Y) → SAVE(input) when requires_grad"
      resource_events_backward:
        - "RELEASE(input) on backward phase end"
      numerics_tags:
        - sigmoid_special_function_4_flops
capabilities:
  - gelu
  - saved_input
priority: 3
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Hendrycks & Gimpel (2016), GELU: https://arxiv.org/abs/1606.08415
- Shazeer (2020), GLU Variants (GeGLU): https://arxiv.org/abs/2002.05202
- Hendrycks, GELU approximations (QuickGELU / FastGELU): https://github.com/hendrycks/GELUs
- PyTorch `Activation.cpp` / `ActivationGeluKernel.cu` (`gelu`, `gelu_backward`): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
- HuggingFace `activations.py` (`GELUActivation`, `NewGELUActivation`, `GELUTanh`, `QuickGELUActivation`): https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Liger-Kernel fused GeGLU: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py
- Liger-Kernel paper (Dai et al., 2024): https://arxiv.org/abs/2410.10989
- Huang & Schlag (2025), xIELU ↔ GELU op-mix comparison: https://arxiv.org/abs/2411.13010
- Zepto modules: `src/zepto/modules/gelu.py`
- Zepto recipes: `src/zepto/analysis/lowering/recipes/gelu.py`
- Zepto discovery rules: `src/zepto/analysis/lowering/implementations/regions/gelu/rules.py`
- Model gaps cross-ref: `docs/model-architecture-gaps-2026-09-14.md` (GELU-tanh, GeGLU)

**Verification date:** 2026-09-14

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 3 | `region/gelu` | §8 tanh variant | standalone_activation | GPT/NewGELU/ViT-tanh path; rules + recipe exist; **RegionImplementation not registered** |
| 3 | `region/gelu_erf` | §8 erf variant | standalone_activation | BERT exact GELU; rules + recipe exist; **not registered** |
| 4 | `region/gelu/quick` | §8 quick variant | standalone_activation | CLIP QuickGELU; recipe exists; no pattern rule yet |
| 4 | `region/geglu/liger` | §3 composable | MLP stack | Liger fuses tanh-GELU+mul; must not double-count with `region/gelu` |
| 5 | `GeGLU` module | §3 | MLP stack | Future — compose `Linear(gate)`, `Linear(up)`, `region/gelu/*`, `Multiply`, `Linear(down)` |

**Open gaps:**
- **`region/gelu*` not registered** — discovery rules and `GeluRecipe` exist; `RegionImplementation` + tests still needed (kernel-implementer).
- **`GELUQuick` module / pattern rule** — QuickGELU recipe constants defined but no Zepto module or `PatternMatchRule` yet.
- **`FastGELUActivation`** — alternate tanh coefficients; numerically close to tanh variant; map to `region/gelu` (tanh) unless a separate tolerance study warrants a leaf.
- **Hub kernel variants** (`GeLU`, `NewGELU`, `QuickGELU`) — no separate Zepto hardware gate; share the same FLOP leaves as ATen paths.
- **Apertus-8B** uses xIELU, not GELU — no Apertus-default numerical trace required; examples use Gemma/ViT dimensions (\(d_{\mathrm{ff}}{=}21504\)).
