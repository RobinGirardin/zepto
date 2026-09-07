# Zepto kernel research: xIELU

**Date:** 2026-09-07
**Scope:** Standalone MLP activation leaf (Apertus decoder FFN); prefill; training + inference; CUDA primary, Python fallback elsewhere
**Context:** Apertus-8B, bf16 primary; kernel-accurate Zepto leaf (Appendix E omits xIELU)

---

## Section 0: Mathematical definition

Huang and Schlag ([2025](https://arxiv.org/abs/2411.13010)) derive xIELU by integrating affine-transformed ELU gradients. Apertus ([Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)):

\[
\mathrm{xIELU}(x) =
\begin{cases}
\alpha_p x^{2} + \beta x & x > 0, \\
\alpha_n \bigl(\mathrm{expm1}(\min(x,\varepsilon)) - x\bigr) + \beta x & x \le 0,
\end{cases}
\]

**I/O shapes:** input pre-activation \(H \in \mathbb{R}^{S \times d_{\mathrm{ff}}}\) (Zepto rank-2; HF/vLLM CUDA often \((B,T,d_{\mathrm{ff}})\)); output \(Z\) same shape. Per-layer scalars: \(\alpha_p, \alpha_n\) (learned via inverse-softplus storage), \(\beta = 0.5\), \(\varepsilon = -10^{-6}\).

**Parameter mapping (training):**
\[
\alpha_p = \mathrm{softplus}(\tilde\alpha_p), \qquad
\alpha_n = \beta + \mathrm{softplus}(\tilde\alpha_n).
\]

**Numerics policies:**
- \(\varepsilon\) clamps the exponential argument on the negative branch (Huang & Schlag §3.4) — affects bytes only when unfused temps materialize `min(x, ε)`.
- Zepto `XIELU` module uses graph inputs `effective_alpha_p` / `effective_alpha_n` to **fold** scalar `softplus` at inference (0 extra FLOPs in the leaf).
- Comparison \(x > 0\) is **not** billed as an arithmetic FLOP (ReLU policy).

**Zepto identity reference:** `src/zepto/modules/xielu.py` — `GreaterThan → positive branch (×3, +) → negative branch (Minimum, Exp, ×2 Subtract, ×2 Multiply, +) → Where` (13 ops).

xIELU is **not** a GLU: one activation between Up and Down GEMMs; \(d_{\mathrm{ff}}\) is scaled \(1.5\times\) vs SwiGLU to match budget (Huang & Schlag §3.5; Apertus §2.4).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | nickjbrowning / rubber-duck-debug CUDA `torch.classes.xielu.XIELU()` ([XIELU repo](https://github.com/nickjbrowning/XIELU)); llama.cpp `ggml_cuda_op_xielu` ([unary.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/unary.cu)) |
| **Identity lowering** | Unfused semantic primitives | Zepto `XIELU`: 13-op chain with full-rank branch temps; HF `XIELUActivation` `torch.where` + `expm1` (computes **both** branches) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/xielu` — **\(8 \lvert H \rvert\)** forward, **\(10 \lvert H \rvert\)** backward; elides branch/`expm1` temps; optional 1-byte sign mask saved |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF eager Python** | [`XIELUActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) | No | — | Any | Both |
| **vLLM** | [`CustomOp` XIELU](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) | Partial | — | CUDA if wheel; else Python | Both |
| **SGLang** | [`BaseFusedOp` XIELU](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/activation.py) | Partial | — | Same as vLLM | Both |
| **nickjbrowning CUDA** | `pip install git+https://github.com/nickjbrowning/XIELU` | Yes | — | CUDA cc ≥ 6.0 | Both (custom autograd) |
| **llama.cpp ggml** | `ggml_cuda_op_xielu` | Yes | — | CUDA GGUF inference | Inference only |
| **Liger Kernel** | — | — | — | — | **No xIELU op** |
| **HF Hub kernels** | — | — | — | Not in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py) | — |
| **Zepto (identity)** | `modules/xielu.py` | No | — | Any | Both |
| **Zepto (registered)** | `region/xielu/reference`, `region/xielu/cuda` | Yes (cost leaf) | — | `any` / `cuda_only` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **MLP activation leaf** — not an attention fusion boundary (A/B/C/D). Fuses the decomposed `Where`/`Exp`/branch chain inside `Up GEMM → xIELU → Down GEMM`.

**Mutually exclusive:**
- `region/xielu` vs identity 13-op chain on the same `XIELU` module invocation — fusion replaces the entire decomposed subgraph.

**Composable:**
- Runs **sequentially** after `region/linear` (Up projection) and before Down GEMM; does not overlap RMSNorm, RoPE, FlashAttention, or Hub kernel patches.
- No Liger or Hub replacement exists for xIELU today.

**Execution constraints:**
- nickjbrowning CUDA expects contiguous tensors; reshapes rank-2 → 3D \((1,S,d_{\mathrm{ff}})\) internally.
- XPU/MPS use Python `where` path only (no fused wheel).
- Capability gate: `requested_capabilities={'fused'}` required for region selection.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert H \rvert = S \cdot d_{\mathrm{ff}}\). Special functions (`Exp`/`expm1`) bill **4 FLOPs** (SiLU-style bucket per §1 / `CONTEXT.md`). Comparisons and `Where` selection: **0 FLOPs**.

### 4.1 Identity lowering (Zepto 13-op chain, eager both-branch semantics)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `GreaterThan` | comparison | 0 | 0 |
| 2–4 | Positive: `Multiply`², `×α_p`, `×β` | 3 mul | 3 | \(3n\) |
| 5 | Positive: `Add` | 1 add | 1 | \(n\) |
| 6 | `Minimum` (clamp) | — | 0 | 0 |
| 7 | `Exp` on clamped | special | 4 | \(4n\) |
| 8–9 | `Subtract` ×2 (`expm1`, `-x`) | 2 | 2 | \(2n\) |
| 10–11 | `Multiply` ×2 (`×α_n`, `×β`) | 2 mul | 2 | \(2n\) |
| 12 | Negative: `Add` | 1 add | 1 | \(n\) |
| 13 | `Where` | select | 0 | 0 |
| **Identity total (both branches evaluated)** | | | | **\(\approx 12n\)** |

Eager HF/vLLM Python evaluates both branches via `torch.where` — same order of magnitude. Full-rank temps (`squared`, `expm1`, `positive`, `negative`, masks) drive **2–3×** peak VRAM vs output alone.

### 4.2 Fused region leaf (kernel-accurate, one branch per element)

**Positive branch** \(\alpha_p x^2 + \beta x\):

| Step | FLOPs |
|------|-------|
| \(x^2\) | 1 mul |
| \(\alpha_p \cdot x^2\) | 1 mul |
| \(\beta \cdot x\) | 1 mul |
| add | 1 add |
| **Subtotal** | **4** |

**Negative branch** \(\alpha_n(\mathrm{expm1}(\min(x,\varepsilon)) - x) + \beta x\):

| Step | FLOPs |
|------|-------|
| clamp + `expm1` (4-FLOP bucket) | 4 |
| \(\mathrm{expm1} - x\) | 1 |
| \(\times \alpha_n\) | 1 |
| \(\beta \cdot x\) | 1 |
| add | 1 |
| **Subtotal** | **8** |

Fused leaf bills the **negative branch as upper bound** (mixed signs per warp):

\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 8 \cdot n = 8 \cdot S \cdot d_{\mathrm{ff}}.
\]

Scalar `softplus` on \(\tilde\alpha_p, \tilde\alpha_n\): \(\approx 18\) FLOPs/layer (\(O(1)\), omitted from leading term). Inference with `effective_alpha_*` inputs: **0**.

### 4.3 Paper-comparable (if different)

Apertus Appendix E **omits** xIELU entirely (MLP leaf = GEMM-only). Label any Appendix-E MLP total that excludes xIELU as **paper-comparable only** — not the Zepto activation leaf.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{xIELU,fwd}} = 8\, S\, d_{\mathrm{ff}}
\]

**Arithmetic intensity (numeric example):** Apertus-8B prefill, \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S \cdot d_{\mathrm{ff}}\) | \(\approx 1.76 \times 10^8\) |
| FLOPs | \(8n \approx 1.41 \times 10^9\) (~1.4B/layer) |
| Ideal 1-pass HBM (read \(H\), write \(Z\)) | \(2n e \approx 672\) MiB |
| AI | \(8/(2e) = 2\) FLOP/byte at bf16 |

xIELU forward is ~**2.5%** of MLP GEMM FLOPs at these shapes but can dominate MLP **HBM traffic** when unfused (branch temps).

---

## Section 5: Backward FLOPs — step-by-step derivation

VJP with saved sign (or full \(H\)): for upstream grad \(g = \partial\mathcal{L}/\partial Z\),

\[
\frac{\partial\mathcal{L}}{\partial H} = g \odot
\begin{cases}
2\alpha_p H + \beta & H > 0 \\
\alpha_n(e^{H} - 1) + \beta & H \le 0
\end{cases}
\]

| Branch | Work | FLOPs/elem |
|--------|------|------------|
| \(H > 0\) | \(2\alpha_p H + \beta\), then \(\odot g\) | **4** |
| \(H \le 0\) | recompute \(e^H\) (4) + affine + \(\odot g\) | **8** |

Parameter grads: masked MAC into two scalars (+ \(\sigma(\tilde\alpha)\) once/layer, \(O(1)\)):

\[
\frac{\partial\mathcal{L}}{\partial\tilde\alpha_p} = \sigma(\tilde\alpha_p)\sum_{H>0} g H^2, \quad
\frac{\partial\mathcal{L}}{\partial\tilde\alpha_n} = \sigma(\tilde\alpha_n)\sum_{H\le 0} g(\mathrm{expm1}(H)-H).
\]

Per-element leading term (neg-branch Jacobian bound + parameter MAC):

\[
\mathrm{FLOPs}_{\mathrm{xIELU,bwd}} = 10 \cdot S \cdot d_{\mathrm{ff}}.
\]

**Do not** use \(16 \cdot S \cdot d_{\mathrm{ff}}\) (GEMM \(2\times\) forward heuristic or forward+Jacobian double-count).

(`requires_grad=False` → backward FLOPs = 0.)

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

Worked unit: one bf16 buffer at \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\) → **336 MiB**.

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Unfused Zepto / HF Python** | `squared`, `expm1`, `positive`, `negative`, masks, output | **672–1008 MiB** (2–3× tile) | — |
| **Fused `region/xielu`** | output \(Z\) only (+ bool sign mask if training) | **336 MiB** + \(\lvert H \rvert\) bytes mask | branch/`expm1`/dual-branch temps |
| **nickjbrowning CUDA / ggml** | output only (inference) | **336 MiB** | on-chip piecewise eval |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| HF Python eager | full input \(H\) + output \(Z\) | \((S,d_{\mathrm{ff}})\) | \(\Theta(n)\) | not modeled as region |
| **Fused `region/xielu`** | **sign mask** (bool) | \((S,d_{\mathrm{ff}})\) | \(\lvert H \rvert\) bytes | `save_sign_mask: true` |
| Alternative recipe | full \(H\) | \((S,d_{\mathrm{ff}})\) | \(n \cdot e\) | future variant |

Sign mask matches `region/relu` mask pattern — Jacobian needs branch id, not output \(Z\) alone.

### 6.3 Resource event chains

**Identity lowering (unfused 13-op chain):**
```
ALLOCATE(squared) → … → ALLOCATE(expm1) → ALLOCATE(positive) → ALLOCATE(negative) → ALLOCATE(Z) → SAVE(Z or H)
```

**Fused region leaf (`region/xielu`):**
```
ALLOCATE(Z) → ALLOCATE(sign_mask) → SAVE(sign_mask)
```

On-chip branch evaluation and `expm1` partials are **elided** — not `ALLOCATE`/`SAVE` in the resource stream.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|---------------|---------------|-----------------|--------------|
| HF Python | transformers | No | ~\(12n\) identity | autograd | Full branch temps | identity |
| nickjbrowning CUDA | XIELU | Yes | \(8n\) | \(10n\) | Output + optional mask | `region/xielu/cuda` |
| llama.cpp ggml | llama.cpp | Yes | \(8n\) | 0 (inference) | In-register dst | — |
| vLLM/SGLang | serving stacks | Partial | \(8n\) when CUDA | \(10n\) | Same as HF/CUDA | `region/xielu/cuda` |
| Zepto fused | `region/xielu` | Yes | \(8n\) | \(10n\) | Sign mask SAVE | `region/xielu/reference` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: xielu
status: registered
recommended_region_ids:
  - impl_id: region/xielu/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: registered
  - impl_id: region/xielu/cuda
    variant: cuda
    hardware_gate: cuda_only
    fusion_boundary: standalone_activation
    status: registered
pattern_rule:
  op_families:
    - greater_than
    - multiply
    - multiply
    - multiply
    - add
    - minimum
    - exp
    - subtract
    - subtract
    - multiply
    - multiply
    - add
    - where
recipe:
  forward_flops: "8 * numel(output)"
  backward_flops: "10 * numel(output) if requires_grad else 0"
  forward_flops_per_element: 8
  backward_flops_per_element: 10
  save_sign_mask: true
  elided_temps:
    - squared
    - expm1
    - positive_branch
    - negative_branch
    - is_positive
    - clamped
  saved_backward:
    - name: sign_mask
      shape: "(S, d_ff) bool"
  resource_events_forward:
    - "ALLOCATE(Z) → ALLOCATE(sign_mask) → SAVE(sign_mask)"
  resource_events_backward:
    - "RELEASE(sign_mask) on backward phase end"
  numerics_tags:
    - expm1_clamp_eps
    - effective_alpha_folded_at_inference
capabilities:
  - fused
  - xielu
priority: 5
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Huang & Schlag (2025), xIELU derivation: https://arxiv.org/abs/2411.13010
- Apertus technical report §2.1 / §2.4: https://arxiv.org/abs/2509.14233
- HuggingFace `XIELUActivation`: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- vLLM XIELU layer: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py
- nickjbrowning CUDA XIELU: https://github.com/nickjbrowning/XIELU
- llama.cpp ggml CUDA xIELU: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/unary.cu
- Zepto domain doc §13: `docs/kernel-implementation.md`
- Zepto module: `src/zepto/modules/xielu.py`
- Registered region: `src/zepto/analysis/lowering/implementations/regions/xielu/`

**Verification date:** 2026-09-07

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| — | `region/xielu/reference` | §8 | standalone | **Registered** — Atto-parity 8/10 FLOP leaf |
| — | `region/xielu/cuda` | §8 | standalone | **Registered** — CUDA hardware gate |
| P4 | `region/xielu` save-\(H\) variant | §6.2 | — | Optional recipe if backend saves full pre-activation instead of sign mask |
| — | Appendix E gap | §4.3 | — | Paper MLP totals omit xIELU; Zepto kernel-accurate estimates should add explicit leaf |

**Open gaps:**
- **Positive-only tensors** bill ~4 FLOPs/elem forward, not 8 — upper bound is conservative.
- **No Liger/Hub fused training kernel** — nickjbrowning CUDA is the only ecosystem fused autograd outside Zepto modeling.
- **Scalar softplus** omitted from leading term; training graphs with raw parameters may want explicit \(+18\) FLOPs/layer annotation.

---

## HANDOFF
- **CONTEXT:** xIELU standalone MLP activation; Apertus-8B prefill; 8 fwd / 10 bwd FLOPs per element; capability-gated fusion.
- **OUTPUT:** `_workspace/research.md`
- **EVIDENCE:** Validator exit code 0; §8 YAML with `region_kind: xielu`, `status: registered`.
- **OPEN:** Positive-branch lower bound; optional save-\(H\) recipe variant; Appendix E omission noted.
- **NEXT:** Architecture delta only if new variants needed; implementation already registered.
