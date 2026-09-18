# Zepto kernel research: Softplus

**Date:** 2026-09-14
**Proposer:** Robin Girardin
**Scope:** Standalone elementwise activation leaf (`log(1 + exp(x))`); scalar parameter transforms (xIELU); per-head attention gates (Laguna-XS, Qwen3-Next); prefill + decode; training + inference; CUDA primary
**Context:** Softplus is missing from Zepto’s registered region set but appears in production stacks as (1) a smooth ReLU surrogate on tensors, (2) a positivity map for learned scalars (xIELU \(\alpha_p, \alpha_n\)), and (3) per-head attention output gates. Default Zepto estimates should use the **fused region leaf**, not a sum of unfused primitives when fusion is selected.

---

## Section 0: Mathematical definition

Softplus ([Dugas et al., 2000](https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing)) is a smooth approximation to ReLU:

\[
\mathrm{softplus}(x) = \log\bigl(1 + e^{x}\bigr).
\]

PyTorch generalizes with \(\beta > 0\) and a numerical **threshold** ([`F.softplus`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html)):

\[
\mathrm{softplus}_\beta(x) =
\begin{cases}
\dfrac{1}{\beta}\log\bigl(1 + e^{\beta x}\bigr) & \beta x \le \text{threshold} \\[4pt]
x & \beta x > \text{threshold}
\end{cases}
\]

Default \(\beta = 1\), `threshold = 20`. The linear branch avoids overflow for large positive inputs; comparisons are **not** billed as arithmetic FLOPs (ReLU policy).

**Derivative (default \(\beta=1\)):**

\[
\frac{\partial\,\mathrm{softplus}}{\partial x} = \sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^{x}}{1 + e^{x}}.
\]

**I/O shapes:** input \(X \in \mathbb{R}^{S \times d}\) (Zepto rank-2; HF/vLLM often \((B, T, d)\) or \((B, T, h)\) for per-head gates); output \(Y\) same shape. Elementwise — no reduction axes.

**Common shape specializations:**

| Use case | Typical shape | Notes |
|----------|---------------|-------|
| MLP / gate activation | \((S, d)\) or \((S, d_{\mathrm{ff}})\) | Same as SiLU/ReLU leaves |
| xIELU scalar params | scalar (0-D) or broadcast | **2 per layer**; \(O(1)\) FLOPs, folded at inference |
| Per-head attention gate | \((S, h)\) or \((B, T, h)\) | Laguna-XS, Qwen3-Next lineage |
| Mish sub-expression | \((S, d)\) | `x * tanh(softplus(x))` — two activations |
| DeepSeek V4 router | \((S, d_{\mathrm{router}})\) | `sqrt(softplus(x))` via `SqrtSoftplusActivation` |

**Numerics policies:**
- Special functions (`exp`, `log`, `log1p`) bill **4 FLOPs per element** each (Zepto coarse bucket; `docs/kernel-implementation.md` §1, xIELU §13 cross-ref).
- Lone `add` / `mul` / `div` = **1 FLOP** each.
- PyTorch CUDA forward uses `log1p(exp(β·x))/β` in the stable branch ([`ActivationSoftplusKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu)).
- PyTorch CUDA backward uses `grad * exp(β·x) / (exp(β·x) + 1)` (sigmoid form) with the same threshold branch.

**Structural vs eager decomposition:**
- **Eager / identity lowering:** `Exp(X)` → temp \(e^{X}\); `Add(1, e^{X})\); `Log` → \(Y\). Three kernel launches, two full-rank intermediates (\(e^{X}\), \(1+e^{X}\)) when unfused.
- **Fused leaf:** one elementwise kernel computes \(\log(1+e^{x})\) with `log1p(exp(·))` in registers — no HBM-resident exponential temp.

**Inverse-softplus parameter storage** (xIELU, not a separate kernel): parameters stored as \(\tilde\alpha = \log(\mathrm{expm1}(\alpha_{\mathrm{target}}))\) so forward applies softplus once per scalar per layer. Inference graphs may fold `softplus(param)` into graph inputs `effective_alpha_*` (0 extra FLOPs).

**Zepto identity reference:** No dedicated `Softplus` module today. Unfused chain would be `Exp → Add → Log` using semantic ops in `src/zepto/semantic/operations/`. xIELU folds scalar softplus into `effective_alpha_p` / `effective_alpha_n` at inference ([`src/zepto/modules/xielu.py`](../../src/zepto/modules/xielu.py)).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch `aten::softplus` / `softplus_backward` ([`ActivationSoftplusKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu)); HF `F.softplus` in xIELU, Mish, `SqrtSoftplusActivation` |
| **Identity lowering** | Unfused semantic chain | `Exp → Add(1) → Log` (3 ops); Zepto `Exp`/`Log` semantic ops exist but bill 0 FLOPs today — identity table uses convention rates |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/softplus` (planned) — **\(9 \lvert Y \rvert\)** forward, **\(7 \lvert Y \rvert\)** backward; elides \(e^{X}\) temp; **SAVE(input)** for training |

**Default estimates use `region/softplus` — not the sum of identity primitive FLOPs when fusion is selected.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **PyTorch ATen** | `F.softplus`, `nn.Softplus`, `aten::softplus` | Yes | standalone | CUDA / CPU / MPS / XPU | Both; autograd saves **input** |
| **PyTorch backward** | `softplus_backward` in [`ActivationSoftplusKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu) | Yes | standalone | CUDA primary | Recomputes sigmoid factor from saved input |
| **HF xIELU** | [`XIELUActivation._xielu_python`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) — two scalar `softplus` per layer | Decomposed eager | scalar leaf | Any | Training; inference may fold into effective alphas |
| **HF Mish** | `input * tanh(F.softplus(input))` | Partial (softplus fused via ATen) | standalone | Any | Both |
| **HF SqrtSoftplus** | [`SqrtSoftplusActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) — `sqrt(softplus(x))` | Partial | router scoring | Any | Both |
| **Laguna-XS gate** | [`modeling_laguna.py`](https://huggingface.co/poolside/Laguna-XS-2.1/blob/main/modeling_laguna.py) — per-head softplus output gate | Decomposed eager | per-head gate | CUDA primary | Both |
| **Qwen3-Next / Qwen3.5 DeltaNet** | Gated DeltaNet decay via softplus-derived parameter | Decomposed eager | recurrent mixer | CUDA primary | Both |
| **Liger Kernel** | — | — | — | — | **No dedicated Softplus Triton kernel** |
| **HF Hub kernels** | — | — | — | — | **Not registered** for Softplus in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py) |
| **Zepto (identity)** | `Exp → Add → Log` (no module) | No | standalone | Any | Both |
| **Zepto (planned)** | `region/softplus` | Yes (cost leaf) | standalone | `any` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **elementwise activation leaf** — not an attention fusion boundary (A/B/C/D). Classify as `standalone_activation` (same family as `region/silu`, `region/relu`, `region/gelu`).

**Mutually exclusive:**
- `region/softplus` vs identity `Exp → Add → Log` on the same invocation — region wins at priority 10 when pattern matches (future `pat-softplus-decomposed`).
- `region/softplus` vs folding scalar softplus into `effective_alpha_*` graph inputs (xIELU inference) — composition choice, not double billing.

**Composable:**
- **xIELU MLP:** two **scalar** softplus ops per layer before `Where`/`expm1` leaf (`region/xielu`). Bill scalars separately (\(O(1)\)) or fold at inference.
- **Per-head attention gate (Laguna-XS):** `Linear(gate) → region/softplus → Multiply(attn_out, gate)` — softplus covers only the nonlinearity; gate projection GEMM and multiply are separate leaves.
- **Mish:** `region/softplus` then `tanh` then `multiply` — three sequential leaves; do not fuse across unless a future `region/mish` leaf is registered.
- **SqrtSoftplus router:** `region/softplus` then `Sqrt` — two leaves; DeepSeek V4 router scoring.
- **DeltaNet decay:** softplus on decay logits inside recurrent mixer — sequential with sigmoid/L2-norm leaves.
- Does not overlap `region/softmax`, `region/gqa/*`, `region/rmsnorm`, etc.

**Execution constraints:**
- Default recipe assumes \(\beta=1\), `threshold=20` (PyTorch defaults). Non-default \(\beta\) scales FLOPs identically per element (one extra `mul` for \(\beta x\) if not folded — negligible vs tensor paths).
- Pattern requires `Log(Add(1, Exp(X)))` or provenance on a `Softplus` module (future).
- `requires_grad=False` → backward FLOPs = 0; no SAVE events.
- Apertus-8B uses xIELU with **two scalar softplus/layer** (negligible vs \(8 S d_{\mathrm{ff}}\) xIELU leaf). Laguna-XS adds **per-head** softplus gates at width \(h \in \{48, 64\}\).

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert Y \rvert\). For xIELU scalars, \(n = 1\) per parameter (2 per layer).

### 4.1 Identity lowering (`Exp → Add → Log`)

Stable form \(\log(1 + e^{x})\):

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Exp(X)` — \(e^{x}\) | special (`exp` bucket) | 4 | \(4n\) |
| 2 | `Add(1, e^{x})` | add | 1 | \(n\) |
| 3 | `Log(1 + e^{x})` | special (`log` bucket) | 4 | \(4n\) |
| **Identity total** | | | | **\(9n\)** |

Two `ALLOCATE` temps in unfused forward (\(e^{X}\) and optionally \(1+e^{X}\)) plus \(Y\).

**Note:** PyTorch fused CUDA uses `log1p(exp(x))` (exp + log1p = 8 special-function FLOPs if `log1p` is bucketed separately). Zepto bills the decomposed \(\ln(1+e^x)\) path at **9** to match xIELU scalar convention (\(c_{\mathrm{sp}} \approx 9\); [`docs/kernel-implementation.md` §13](../../docs/kernel-implementation.md)).

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused \(e^{x}\) in registers | special | 4 | \(4n\) |
| 2 | Fused \(1 + e^{x}\) | add | 1 | \(n\) |
| 3 | Fused \(\log(\cdot)\) via `log1p` | special | 4 | \(4n\) |
| **Fused total** | | | | **\(9n\)** |

Forward FLOPs are **identical** for identity and fused paths; fusion elides HBM traffic for \(e^{X}\), not forward arithmetic.

**Threshold branch** (\(\beta x > 20\)): returns \(x\) with 0 special-function FLOPs. For typical initialized gates and xIELU scalars the stable branch runs; Zepto bills **9** uniformly (kernel-accurate upper bound for mixed-sign tensors).

### 4.3 Paper-comparable (if different)

Dugas et al. define softplus as a smooth ReLU surrogate without separate FLOP accounting. Appendix E for Apertus **omits** activations (MLP = GEMM-only); that omission is not a lower softplus constant.

xIELU scalar billing cross-check: two softplus/layer \(\approx 18\) forward FLOPs (\(2 \times 9\)), correctly dropped from the \(O(S d_{\mathrm{ff}})\) leading term ([`docs/kernel/xielu.md`](../../docs/kernel/xielu.md)).

**Closed form (kernel-accurate, \(\beta=1\)):**
\[
\mathrm{FLOPs}_{\mathrm{softplus,fwd}} = 9 \cdot n.
\]

**Arithmetic intensity (numeric example A — tensor gate):** Laguna-style per-head gate, \(S{=}8192\), \(h{=}48\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S \cdot h\) | \(\approx 3.93 \times 10^5\) |
| FLOPs | \(9n \approx 3.5 \times 10^6\) |
| Min HBM bytes (read \(X\) + write \(Y\), fused) | \(2 n e \approx 1.5\) MiB |
| Arithmetic intensity | \(\approx 0.23\) FLOP/byte — **memory-bound** |

**Arithmetic intensity (numeric example B — xIELU scalars):** Apertus-8B, \(L{=}32\) layers:

| Quantity | Value |
|----------|-------|
| Forward FLOPs | \(2 \times 9 \times L = 576\) |
| Share of xIELU leaf (\(8 S d_{\mathrm{ff}} L\) at \(S{=}8192\)) | \(\ll 0.001\%\) — correctly negligible |

---

## Section 5: Backward FLOPs — step-by-step derivation

Softplus VJP (default \(\beta=1\)):

\[
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \sigma(x) = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{e^{x}}{1 + e^{x}}.
\]

PyTorch autograd saves **input** \(X\) (not \(Y\) nor \(\sigma(X)\)) and recomputes the sigmoid factor in `softplus_backward` ([`ActivationSoftplusKernel.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu)).

### 5.1 Identity lowering (unfused `Exp → Add → Log`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Log` VJP: `grad / (1+e^x)` | div | 1 | \(n\) |
| 2 | Recompute `Exp(x)` for chain | special | 4 | \(4n\) |
| 3 | `Exp` VJP: `grad_inter * exp(x)` | mul | 1 | \(n\) |
| 4 | `Add` VJP: pass-through | — | 0 | 0 |
| **Identity backward total** | | | | **\(6n\)**† |

†Unfused semantic backward undercounts if `Log`/`Exp` ops bill 0 in Zepto today; fused leaf is authoritative.

### 5.2 Fused region leaf (kernel-accurate)

Matches PyTorch CUDA backward kernel (`grad * exp(x) / (exp(x) + 1)`):

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(z = e^{x}\) | special | 4 | \(4n\) |
| 2 | \(z + 1\) | add | 1 | \(n\) |
| 3 | \(z / (z + 1)\) (sigmoid factor) | div | 1 | \(n\) |
| 4 | `grad_out * sigmoid` | mul | 1 | \(n\) |
| **Fused backward total** | | | | **\(7n\)** |

**Alternative sigmoid-bucket billing:** recompute \(\sigma(x)\) as one special function (4) + grad mul (1) = **5 FLOPs/element**. xIELU scalar training adds **\(+14\)** backward FLOPs for two scalars (\(7\) each), matching the exp/add/div/mul decomposition above — use **7** as the Zepto leaf constant.

**Threshold branch** (\(\beta x > 20\)): gradient passes through (`grad_out`); **0** extra arithmetic — ignored in uniform per-element billing.

`requires_grad=False` → backward FLOPs = **0**.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{softplus,bwd}} = \begin{cases} 7 \cdot n & \text{if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

**xIELU training add-on:** \(+18\) forward / \(+14\) backward FLOPs per layer for two scalar softplus ops (negligible vs tensor xIELU leaf).

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Identity (unfused)** | \(e^{X}\), \(1+e^{X}\) (optional), \(Y\) | up to \(2 n e\) | — |
| **Fused `region/softplus`** | \(Y\) only | \(n e\) | \(e^{X}\) (\(n e\)) |

Input \(X\) is a boundary edge (not re-allocated by the region).

**Scalar xIELU path:** no tensor temps; two scalar outputs live in registers / parameter cache.

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| PyTorch `aten::softplus` autograd | input \(X\) | same as \(Y\) | \(n e\) | `save_for_backward(input)` |
| **`region/softplus`** | input \(X\) | \((S,d)\) or \((S,h)\) | \(n e\) | `save_input: true` when `requires_grad` |
| Unfused identity | `Exp` saves output; `Log` saves input | — | up to \(2 n e\) | redundant if chain not fused |

Fused softplus does **not** materialize \(e^{X}\) or \(\sigma(X)\) to HBM; backward recomputes sigmoid factor from saved \(X\) (same strategy as PyTorch fused backward).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(exp_X) → SAVE(input) on exp → ALLOCATE(one_plus_exp) → ALLOCATE(Y) → SAVE(input) on log
```

**Fused region leaf (`region/softplus`):**
```
ALLOCATE(Y) → SAVE(input)
```

Backward phase end: `RELEASE(input)` on the saved input edge.

**Scalar xIELU (folded at inference):**
```
(no ALLOCATE — effective_alpha_* are graph inputs)
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| PyTorch ATen | torch | Yes | standalone | \(9n\) | \(7n\) | SAVE(input) | — |
| HF xIELU scalars | transformers | Decomposed | scalar | \(9\) each | \(7\) each | scalar registers | fold at inference |
| HF Mish / SqrtSoftplus | transformers | Partial | standalone | \(9n\) slice | \(7n\) slice | SAVE(input) | — |
| Laguna-XS gate | poolside | Decomposed | per-head | \(9n\) | \(7n\) | SAVE(input) | — |
| Liger | liger-kernel | — | — | — | — | — | none |
| Zepto identity | Exp+Add+Log | No | standalone | \(9n\) | \(6n\)† | Multiple SAVEs | — |
| Zepto fused (planned) | `region/softplus` | Yes | standalone | \(9n\) | \(7n\) | SAVE(input) | to_implement |

†Identity backward undercounts when semantic ops bill 0; fused leaf is authoritative.

---

## Section 8: Zepto implementation spec

```yaml
region_kind: softplus
recommended_region_ids:
  - id: region/softplus
    variant: default
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    capabilities:
      - softplus
      - saved_input
    pattern_rule:
      op_families:
        - exp
        - add
        - log
      edge_constraints:
        - log_add_exp_chain  # Log(Add(1, Exp(X)))
      constraints:
        - softplus_decomposed_chain
    recipe:
      forward_flops: "9 * numel(output)"
      backward_flops: "7 * numel(output) if requires_grad else 0"
      forward_flops_per_element: 9
      backward_flops_per_element: 7
      beta: 1
      threshold: 20
      save_input: true
      elided_temps:
        - exp_output  # e^X — computed in registers, not ALLOCATE'd
      saved_backward:
        - name: input
          shape: "(S, d) or (S, h) or scalar"
      resource_events_forward:
        - "ALLOCATE(Y) → SAVE(input) when requires_grad"
      resource_events_backward:
        - "RELEASE(input) on backward phase end"
      numerics_tags:
        - log1p_exp_stable
        - threshold_linear_branch
        - exp_log_special_function_4_flops_each
    priority: 10
  - id: region/softplus/scalar
    variant: xielu_scalar
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    capabilities:
      - softplus
      - scalar
    recipe:
      forward_flops: "9 * num_scalar_inputs"
      backward_flops: "7 * num_scalar_inputs if requires_grad else 0"
      forward_flops_per_element: 9
      backward_flops_per_element: 7
      save_input: false
      elided_temps: []
      saved_backward: []
      resource_events_forward: []
      numerics_tags:
        - fold_into_effective_alpha_at_inference
    priority: 5
capabilities:
  - softplus
  - saved_input
priority: 6
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Dugas, Bengio, Belisle, Nadeau, LeC (2000), Incorporating second-order functional knowledge for better option pricing (Softplus): https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing
- PyTorch `ActivationSoftplusKernel.cu` (`softplus`, `softplus_backward`): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu
- PyTorch `Activation.cpp` (meta dispatch): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- PyTorch `F.softplus` documentation: https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
- HuggingFace `activations.py` (xIELU scalar softplus, Mish, SqrtSoftplus): https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Laguna-XS modeling (per-head softplus gate): https://huggingface.co/poolside/Laguna-XS-2.1/blob/main/modeling_laguna.py
- Qwen3.5 modular decoder (DeltaNet softplus decay): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py
- xIELU paper (Huang & Schlag, 2025): https://arxiv.org/abs/2411.13010
- Apertus technical report (xIELU §2.4): https://arxiv.org/abs/2509.14233
- Zepto xIELU module: `src/zepto/modules/xielu.py`
- Zepto gaps survey: `docs/model-architecture-gaps-2026-09-14.md`
- Cross-ref: `docs/kernel-implementation.md` §13 (xIELU scalar softplus \(c_{\mathrm{sp}} \approx 9\) fwd / 7 bwd)

**Verification date:** 2026-09-14

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 6 | `region/softplus` | §8 | standalone_activation | **Missing** — blocks Laguna-XS/Qwen3-Next per-head gates, Mish/SqrtSoftplus decomposition, and explicit scalar billing for xIELU training graphs |
| 6 | `Softplus` module | §0 | standalone | Future — `src/zepto/modules/softplus.py` for identity lowering + region pattern match |
| 7 | `region/mish` | §3 composable | standalone | Future — fuse `softplus + tanh + mul` when Mish modules are modeled |
| 7 | `region/sqrt_softplus` | §3 composable | router | Future — DeepSeek V4 `SqrtSoftplusActivation` |

**Open gaps:**
- **`Softplus` semantic op** not registered — FLOPs fall through identity chain with `Exp`/`Log` billing at 0 today.
- **Non-default \(\beta\)/threshold** — recipe assumes PyTorch defaults; exotic configs need a variant flag.
- **Hub / Liger** — no fused Softplus kernel to register as a hardware-gated variant; ATen path is the reference.
- **xIELU inference folding** — document whether scalar softplus is elided via `effective_alpha_*` inputs (0 FLOPs) vs explicit `region/softplus/scalar` (9 FLOPs) in training-only estimates.
