# Zepto kernel research: SiLU (Swish)

**Date:** 2026-09-14
**Proposer:** Robin Girardin
**Scope:** Standalone elementwise activation leaf (`x · σ(x)`); prefill + decode; training + inference; CUDA primary; SwiGLU gate branch cross-ref
**Context:** SiLU is the gating nonlinearity in SwiGLU FFNs (Llama, Granite, Qwen, …). Apertus uses xIELU instead, but SiLU is a first-class Zepto leaf for gated-MLP cost modeling. Default Zepto estimates use the **fused region leaf**, not a sum of unfused primitives.

---

## Section 0: Mathematical definition

SiLU (Sigmoid Linear Unit), also called **Swish-1** when \(\beta=1\) ([Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)):

\[
\mathrm{SiLU}(x) = x \cdot \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}.
\]

**I/O shapes:** input \(X \in \mathbb{R}^{S \times d}\) (Zepto rank-2; HF/vLLM often \((B, T, d)\)); output \(Y\) same shape. Elementwise — no reduction axes.

**Numerics policies:**
- Special function \(\sigma\) (via `exp`) bills **4 FLOPs per element** (Zepto / Atto coarse bucket; see `docs/kernel-implementation.md` §1 and `src/zepto/semantic/operations/sigmoid.py`).
- Comparisons and branch selection: **0 FLOPs** (ReLU policy).
- PyTorch `aten::silu` and `F.silu` use a single fused CUDA kernel on GPU; eager decomposition is `sigmoid(x) * x` ([`Activation.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp)).

**Structural vs eager decomposition:**
- **Eager / identity lowering:** `Sigmoid(X)` → temp \(\sigma(X)\); `Multiply(X, σ(X))` → \(Y\). Two kernel launches, one full-rank intermediate.
- **Fused leaf:** one elementwise kernel computes \(Y = X \cdot \sigma(X)\) with \(\sigma\) in registers — no HBM-resident \(\sigma(X)\) tensor.

**SwiGLU composition (parent module, not this leaf):**
\[
\mathrm{SwiGLU}(x) = \mathrm{SiLU}(W_g x) \odot (W_u x)
\]
([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). Zepto models the **SiLU gate** as `region/silu` and the gate×up product as a separate `Multiply`; Liger fuses the entire SwiGLU post-GEMM chain.

**Zepto identity reference:** `src/zepto/modules/silu.py` — `Sigmoid(x)` then `Multiply(x, sig)`.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch `aten::silu` / `silu_backward` ([`Activation.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp)); Liger fused SwiGLU (SiLU inside) ([Liger-Kernel](https://github.com/linkedin/Liger-Kernel)); HF Hub `@use_kernel_forward_from_hub("SiLU")` |
| **Identity lowering** | Unfused semantic chain | Zepto `Sigmoid → Multiply` (2 ops); HF `nn.functional.silu` dispatches fused ATen, but semantic graph decomposes to sigmoid + mul |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/silu` — **\(5 \lvert Y \rvert\)** forward, **\(8 \lvert Y \rvert\)** backward; elides \(\sigma(X)\) temp; **SAVE(input)** for training |

**Default estimates use `region/silu` — not the sum of identity `Sigmoid` + `Multiply` FLOPs when fusion is selected.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **PyTorch ATen** | `F.silu`, `nn.SiLU`, `aten::silu` | Yes | standalone | CUDA / CPU / MPS / XPU | Both; autograd saves **input** |
| **PyTorch decomposed backward** | `math_silu_backward` in [`Activation.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp) | Reference formula | standalone | Any | Recomputes \(\sigma\) from saved input |
| **HF Transformers** | [`SiLUActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) → `F.silu` | Yes (ATen) | standalone | Any | Both |
| **HF Hub kernels** | `@use_kernel_forward_from_hub("SiLU")` on `SiLUActivation` | Yes | standalone | Hub-routed when `use_kernels=True` | Both |
| **Liger SwiGLU** | [`liger-kernel`](https://github.com/linkedin/Liger-Kernel) / TRL `use_liger_kernel=True` | Yes (SiLU inside SwiGLU) | MLP gate branch | CUDA (Triton); CuTe DSL optional | Training-focused |
| **vLLM / SGLang MLP** | Model-specific SwiGLU modules | Partial | MLP stack | CUDA primary | Both |
| **Zepto (identity)** | `modules/silu.py` → `sigmoid` + `multiply` | No | standalone | Any | Both |
| **Zepto (registered)** | `region/silu` | Yes (cost leaf) | standalone | `any` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **elementwise activation leaf** — not an attention fusion boundary (A/B/C/D). Classify as `standalone_activation` (same family as `region/relu`, `region/xielu`).

**Mutually exclusive:**
- `region/silu` vs identity `Sigmoid → Multiply` on the same `SiLU` module invocation — region wins at priority 10 when pattern matches (`pat-silu-decomposed`, `prov-silu`).

**Composable:**
- **SwiGLU FFN:** `Linear(gate) → region/silu → Multiply(gate_act, up)` → `Linear(down)`. SiLU region covers only the gate nonlinearity; gate×up multiply and three GEMMs are separate leaves.
- **Liger SwiGLU** replaces the post-GEMM SiLU+mul chain with one Triton kernel — Zepto should not double-count `region/silu` when a future `region/swiglu/liger` leaf is selected.
- Does not overlap attention regions (`region/softmax`, `region/gqa/*`, …).
- Liger patches (RMSNorm, SwiGLU, CE) compose with FlashAttention Hub kernels per [TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub).

**Execution constraints:**
- Pattern requires `Multiply(X, Sigmoid(X))` with shared input edge (`silu_mul_x_sigmoid_x` constraint).
- Provenance rule matches `SiLU` module with contiguous in-graph order.
- `requires_grad=False` → backward FLOPs = 0; no SAVE events.
- Apertus-8B does **not** use SiLU in its FFN (xIELU instead); SiLU costing applies to Llama/Granite/Qwen-style stacks at \(d_{\mathrm{ff}}\) width.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert Y \rvert = S \cdot d\) (or \(S \cdot d_{\mathrm{ff}}\) when modeling an MLP gate projection output).

### 4.1 Identity lowering (`Sigmoid → Multiply`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Sigmoid(X)` — \(\sigma(x)=1/(1+e^{-x})\) | special (`exp` bucket) | 4 | \(4n\) |
| 2 | `Multiply(X, σ(X))` | mul | 1 | \(n\) |
| **Identity total** | | | | **\(5n\)** |

Two `ALLOCATE` temps in unfused forward (\(\sigma(X)\) and \(Y\)).

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused \(\sigma(x)\) in registers | special | 4 | \(4n\) |
| 2 | Fused \(x \cdot \sigma(x)\) | mul | 1 | \(n\) |
| **Fused total** | | | | **\(5n\)** |

Forward FLOPs are **identical** for identity and fused paths; fusion elides HBM traffic for \(\sigma(X)\), not forward arithmetic.

### 4.3 Paper-comparable (if different)

Ramachandran et al. define Swish as \(x \cdot \sigma(x)\) with the same op mix (one sigmoid/exp, one multiply). **No cheaper paper-comparable forward formula** — Appendix E for Apertus **omits** activations entirely (MLP = GEMM-only); that omission is not a lower SiLU constant.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{SiLU,fwd}} = 5 \cdot n = 5 \cdot S \cdot d.
\]

**Arithmetic intensity (numeric example):** Llama-style prefill, \(S{=}8192\), \(d_{\mathrm{ff}}{=}14336\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S \cdot d_{\mathrm{ff}}\) | \(\approx 1.18 \times 10^8\) |
| FLOPs | \(5n \approx 5.9 \times 10^8\) |
| Min HBM bytes (read \(X\) + write \(Y\), fused) | \(2 n e \approx 452\) MiB |
| Arithmetic intensity | \(\approx 0.31\) FLOP/byte — **memory-bound** |

At Apertus-scale \(d_{\mathrm{ff}}{=}21504\): \(5n \approx 8.8 \times 10^8\) FLOPs/layer — ~**1.5%** of Up+Down GEMM FLOPs (\(\approx 57\)B/layer at \(S{=}8192\)), comparable to xIELU’s ~2.5% share.

---

## Section 5: Backward FLOPs — step-by-step derivation

SiLU derivative ([`math_silu_backward`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp)):

\[
\frac{\partial \mathrm{SiLU}}{\partial x} = \sigma(x)\,\bigl(1 + x\,(1 - \sigma(x))\bigr).
\]

\[
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \sigma(x)\,\bigl(1 + x\,(1 - \sigma(x))\bigr).
\]

PyTorch autograd saves **input** \(X\) (not \(\sigma(X)\) nor \(Y\)) and recomputes \(\sigma\) in the backward kernel.

### 5.1 Identity lowering (unfused `Sigmoid → Multiply`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Multiply` VJP, left: `grad_out * σ` | mul | 1 | \(n\) |
| 2 | `Multiply` VJP, right: `grad_out * x` | mul | 1 | \(n\) |
| 3 | `Sigmoid` VJP on \(\sigma\) path (recompute \(\sigma\)) | special | 4 | \(4n\) |
| **Identity backward total** | | | | **\(6n\)** |

Note: Zepto `Sigmoid.backward_flops` bills recompute only (4 FLOPs); the multiply chain adds 2 FLOPs. Full sigmoid VJP arithmetic is folded into the fused leaf below.

### 5.2 Fused region leaf (kernel-accurate)

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(\sigma(x)\) from saved \(x\) | special | 4 | \(4n\) |
| 2 | \((1 - \sigma)\) | sub | 1 | \(n\) |
| 3 | \(x \cdot (1 - \sigma)\) | mul | 1 | \(n\) |
| 4 | \(1 + x(1-\sigma)\) then \(\sigma \cdot (\cdots)\) (fused VJP factor) | add + mul | 2 | \(2n\) |
| 5 | `grad_out * factor` | mul | 1 | \(n\) |
| **Fused backward total** | | | | **\(8n\)** |

Consolidated recipe form (matches `SiLURecipe`): **4** (sigmoid recompute) + **3** (fused VJP) + **1** (grad multiply) = **8 FLOPs/element**.

`requires_grad=False` → backward FLOPs = **0**.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{SiLU,bwd}} = \begin{cases} 8 \cdot n & \text{if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

**Common mistake:** Do **not** use \(2 \times 5 = 10\) (GEMM-style “backward ≈ 2× forward”) or \(5 + 8 = 13\) (double-counting exp inside VJP). Atto sibling framework documents **5 forward / 8 backward** ([`docs/kernel-implementation.md`](../docs/kernel-implementation.md) §13 cross-ref).

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Identity (unfused)** | \(\sigma(X)\), \(Y\) | \(2 n e\) | — |
| **Fused `region/silu`** | \(Y\) only | \(n e\) | \(\sigma(X)\) (\(n e\)) |

Input \(X\) is a boundary edge (not re-allocated by the region).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| PyTorch `aten::silu` autograd | input \(X\) | \((S,d)\) | \(n e\) | `save_for_backward(input)` |
| **`region/silu`** | input \(X\) | \((S,d)\) | \(n e\) | `save_input: true` when `requires_grad` |
| Unfused identity | `Sigmoid` saves \(X\); `Multiply` saves \(X, \sigma(X)\) | — | up to \(2 n e + n\) | redundant if chain not fused |

Fused SiLU does **not** materialize \(\sigma(X)\) to HBM; backward recomputes \(\sigma\) from saved \(X\) (same strategy as PyTorch fused `silu_backward`).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(σ(X)) → SAVE(input) on sigmoid → ALLOCATE(Y) → SAVE(left, right) on multiply
```

**Fused region leaf (`region/silu`):**
```
ALLOCATE(Y) → SAVE(input)
```

Backward phase end: `RELEASE(input)` on the saved input edge.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| PyTorch ATen | torch | Yes | standalone | \(5n\) | \(8n\) | SAVE(input) | — |
| HF eager | transformers | Yes (ATen) | standalone | \(5n\) | \(8n\) | SAVE(input) | — |
| Liger SwiGLU | liger-kernel | Yes (SiLU in SwiGLU) | MLP gate | \(5n\) gate slice* | \(8n\) gate slice* | Elide gate/up temps | future `region/swiglu` |
| Zepto identity | sigmoid+multiply | No | standalone | \(5n\) | \(6n\)† | Multiple SAVEs | — |
| Zepto fused | `region/silu` | Yes | standalone | \(5n\) | \(8n\) | SAVE(input) | registered |

\*Gate-branch slice only; full SwiGLU adds two GEMMs and one elementwise mul.  
†Identity backward undercounts full sigmoid VJP per semantic op billing; fused leaf is authoritative.

---

## Section 8: Zepto implementation spec

```yaml
region_kind: silu
recommended_region_ids:
  - id: region/silu
    variant: default
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: registered
    capabilities:
      - silu
      - saved_input
pattern_rule:
  op_families:
    - sigmoid
    - multiply
  edge_constraints:
    - output_to_second_input  # σ(X) → multiply right input
  constraints:
    - silu_mul_x_sigmoid_x
recipe:
  forward_flops: "5 * numel(output)"
  backward_flops: "8 * numel(output) if requires_grad else 0"
  forward_flops_per_element: 5
  backward_flops_per_element: 8
  save_input: true
  elided_temps:
    - sigmoid_output  # σ(X) — computed in registers, not ALLOCATE'd
  saved_backward:
    - name: input
      shape: "(S, d)"
  resource_events_forward:
    - "ALLOCATE(Y) → SAVE(input) when requires_grad"
  resource_events_backward:
    - "RELEASE(input) on backward phase end"
  numerics_tags:
    - sigmoid_special_function_4_flops
    - comparison_not_billed
capabilities:
  - silu
  - saved_input
priority: 10
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Ramachandran, Zoph, Le (2017), Swish / SiLU: https://arxiv.org/abs/1710.05941
- Shazeer (2020), GLU Variants (SwiGLU): https://arxiv.org/abs/2002.05202
- PyTorch `Activation.cpp` (`silu`, `silu_backward`, `math_silu_backward`): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- HuggingFace `SiLUActivation` / Hub kernel hook: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Liger-Kernel (fused SwiGLU): https://github.com/linkedin/Liger-Kernel
- Liger-Kernel paper (Dai et al., 2024): https://arxiv.org/abs/2410.10989
- TRL Liger integration: https://huggingface.co/docs/trl/en/liger_kernel_integration
- Zepto module: `src/zepto/modules/silu.py`
- Zepto region: `src/zepto/analysis/lowering/implementations/regions/silu/`
- Zepto recipe: `src/zepto/analysis/lowering/recipes/silu.py`
- Cross-ref: `docs/kernel-implementation.md` §1 (special-function bucket), §13 (xIELU ↔ SiLU FLOP parity)

**Verification date:** 2026-09-14

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| — | `region/silu` | §8 | standalone_activation | **Registered** — fused SiLU leaf for SwiGLU gate branches and standalone `SiLU` modules |
| 4 | `region/swiglu/liger` | §3 composable | MLP stack | Future — Liger fuses SiLU+mul+GEMM context; must not double-count with `region/silu` |
| 5 | `SwiGLU` module | §3 | MLP stack | Future — compose `Linear(gate)`, `Linear(up)`, `region/silu`, `Multiply`, `Linear(down)` |

**Open gaps:**
- **`region/swiglu` parent region** not yet registered — callers composing gate/up GEMMs + SiLU + mul need manual graph assembly today.
- **Hub `SiLU` kernel variant** — no separate Zepto hardware gate; ATen/Hub paths share the same 5/8 FLOP leaf.
- **GPT-OSS clamped SiLU expert** (`SiLU(1.702×gate)`) — variant activation; not covered by default `region/silu` pattern (see `docs/model-architecture-gaps-2026-09-14.md`).
