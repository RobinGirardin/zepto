# Zepto kernel research: Squared ReLU (ReLU²)

**Date:** 2026-09-14
**Proposer:** Robin Girardin
**Scope:** Standalone elementwise activation leaf \(\mathrm{ReLU}(x)^2\); identity chain `Maximum → Multiply` (self-square); prefill + decode; training + inference; CUDA primary; two-projection MLP cross-ref (Primer, Nemotron, Persimmon, …)
**Context:** Squared ReLU is the MLP nonlinearity in Primer ([So et al., 2021](https://arxiv.org/abs/2109.08668)) and in several HF configs (`hidden_act: "relu2"` — Nemotron, Nemotron-H, Persimmon, BitNet, Arcee, Jais2, NanoChat, Fuyu). Zepto has `ReLU` and `Multiply` primitives but **no** fused `region/squared_relu` or `ReLUSquared` module yet (`docs/model-architecture-gaps-2026-09-14.md`). Default Zepto estimates should use the **fused region leaf**, not a sum of unfused primitives when fusion is selected.

---

## Section 0: Mathematical definition

Squared ReLU (ReLU², config key `relu2` in HuggingFace Transformers):

\[
\mathrm{ReLU}^2(x) = \bigl(\max(x, 0)\bigr)^2 = \begin{cases} x^2 & x > 0 \\ 0 & x \le 0 \end{cases}
\]

Equivalently: \(\mathrm{ReLU}^2(x) = \mathrm{ReLU}(x) \cdot \mathrm{ReLU}(x)\).

**Primary motivation:** Primer’s architecture search found that squaring ReLU activations in the FFN reduces training cost versus standard ReLU/GELU at scale ([So et al., 2021](https://arxiv.org/abs/2109.08668)).

**I/O shapes:** input \(X \in \mathbb{R}^{S \times d}\) (Zepto rank-2; HF/vLLM often \((B, T, d)\) or \((B, T, d_{\mathrm{ff}})\) after the up-projection); output \(Y\) same shape. Elementwise — no reduction axes.

**Numerics policies:**
- Comparison \(x > 0\) is **not** billed as an arithmetic FLOP (Zepto / ReLU policy; `docs/kernel-implementation.md` §1).
- `max(x, 0)` bills **1 FLOP per element** (`Maximum` primitive).
- Self-square via `Multiply(a, a)` or `torch.square` bills **1 FLOP per element** (single mul).
- Subgradient at \(x = 0\) is 0 (PyTorch `aten::relu` convention; product with zero output enforces zero gradient).

**Structural vs eager decomposition:**
- **Eager / HF Python:** `F.relu(x)` → temp \(A = \mathrm{ReLU}(X)\); `torch.square(A)` → \(Y\). Two kernel launches, one full-rank intermediate ([`ReLUSquaredActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)).
- **Fused leaf:** one elementwise kernel computes \(Y = \max(0, x)^2\) with the ReLU gate and square in registers — no HBM-resident \(\mathrm{ReLU}(X)\) tensor.

**Two-projection MLP composition (parent module, not this leaf):**
\[
\mathrm{MLP}(x) = W_2 \,\mathrm{ReLU}^2(W_1 x + b_1) + b_2
\]
(Primer / Nemotron-style dense FFN). Zepto models **ReLU²** as `region/squared_relu` and the two GEMMs as separate `region/linear` leaves.

**Zepto identity reference (proposed):** decomposed chain matching HF eager — `Maximum(x, 0)` then `Multiply(relu_out, relu_out)`. No dedicated `ReLUSquared` module exists in `src/zepto/modules/` today; only `src/zepto/modules/relu.py` (`ReLU` → single `Maximum`).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | No dedicated `aten::relu_squared`; PyTorch executes **fused elementwise** `relu` + `square` kernels sequentially ([`Activation.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp)); HF `ReLUSquaredActivation` → `F.relu` + `torch.square` |
| **Identity lowering** | Unfused semantic chain | Zepto `Maximum(x, 0) → Multiply(relu_out, relu_out)` (2 ops); HF eager two-launch path |
| **Zepto fused region** | Kernel-accurate cost leaf | **`region/squared_relu`** (proposed) — **\(2 \lvert Y \rvert\)** forward, **\(3 \lvert Y \rvert\)** backward; elides \(\mathrm{ReLU}(X)\) temp; **SAVE(input)** for training |

**Default estimates use the fused region leaf — not the sum of identity primitive FLOPs when fusion is selected.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF `ReLUSquaredActivation`** | [`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py) — `relu2` in `ACT2FN` | No (2 eager ops) | standalone | Any | Both; autograd saves **relu input** + **square input** (\(\mathrm{ReLU}(X)\)) |
| **PyTorch `F.relu` + `torch.square`** | [`Activation.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp), [`TensorFactories.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp) | Per-op fused ATen | standalone | CUDA / CPU / MPS / XPU | Both |
| **Primer (T5 codebase)** | [google-research/primer](https://github.com/google-research/google-research/tree/master/primer) — `hidden_act="relu2"` | Framework-dependent | MLP stack | TPU/GPU | Training |
| **Nemotron / Nemotron-H** | [`modeling_nemotron.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron/modeling_nemotron.py), [`configuration_nemotron.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron/configuration_nemotron.py) (`hidden_act: relu2`) | Via HF activation | MLP stack | CUDA primary | Both |
| **Persimmon, BitNet, Arcee, Jais2, NanoChat, Fuyu** | HF configs with `hidden_act: "relu2"` | Via HF activation | MLP stack | Model-dependent | Both |
| **Liger Kernel** | [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | **No dedicated ReLU² op** | — | CUDA (Triton) | Training-focused |
| **HF Hub kernels** | `@use_kernel_forward_from_hub` | **No `ReLU²` Hub entry** (unlike SiLU/GELU) | — | — | — |
| **vLLM / SGLang** | Generic MLP + `ACT2FN` | Partial (ATen per op) | MLP stack | CUDA primary | Both |
| **Zepto (identity)** | `ReLU` + `Multiply` (manual chain) | No | standalone | Any | Both |
| **Zepto (proposed region)** | `region/squared_relu` | Yes (cost leaf) | standalone | `any` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **elementwise activation leaf** — not an attention fusion boundary (A/B/C/D). Classify as `standalone_activation` (same family as `region/relu`, `region/silu`, `region/gelu`).

**Mutually exclusive:**
- `region/squared_relu` vs identity `Maximum → Multiply(relu, relu)` on the same activation invocation — region wins at priority 10 when pattern matches (proposed `pat-squared-relu-decomposed`).
- `region/squared_relu` vs standalone `region/relu` — different op families; ReLU² pattern requires the self-multiply after ReLU.
- `region/squared_relu` vs `region/relu` + separate square `Multiply` counted twice — fusion must replace the full two-op chain.

**Composable:**
- **Two-projection MLP:** `Linear(up) → region/squared_relu → Linear(down)`. ReLU² region covers only the nonlinearity; two GEMMs are separate leaves.
- **Primer / Nemotron hybrid stacks:** ReLU² sits between attention/Mamba blocks and downstream norms; no overlap with `region/softmax`, `region/gqa/*`, `region/rmsnorm`.
- Does **not** compose with SwiGLU/GeGLU gate branches (those use SiLU/GELU, not ReLU²).

**Execution constraints:**
- Pattern requires `Multiply(A, A)` where \(A = \mathrm{ReLU}(X)\) and \(A\) is the sole output of `Maximum(X, 0)` with shared input edge to the multiply (proposed constraint `squared_relu_mul_relu_relu`).
- Right operand of `Maximum` must be provably zero (`PatternConstraint(kind="right_operand_is_zero")`).
- Provenance rule (proposed) matches a future `ReLUSquared` module with contiguous in-graph order.
- `requires_grad=False` → backward FLOPs = 0; no SAVE events.
- Apertus-8B uses **xIELU**, not ReLU²; costing applies to Primer/Nemotron/Persimmon-style stacks at width \(d_{\mathrm{ff}}\).

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert Y \rvert = S \cdot d\) (or \(S \cdot d_{\mathrm{ff}}\) for an MLP intermediate).

Comparisons (\(x > 0\)): **0 FLOPs**.

### 4.1 Identity lowering (`Maximum → Multiply`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Maximum(X, 0)` — \(\mathrm{ReLU}(x)\) | max | 1 | \(n\) |
| 2 | `Multiply(A, A)` — \(A^2\) | mul | 1 | \(n\) |
| **Identity total** | | | | **\(2n\)** |

Unfused forward allocates up to **2** full-rank temps (\(\mathrm{ReLU}(X)\) and \(Y\)) beyond the input boundary edge.

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused \(\max(0, x)\) in registers | max | 1 | \(n\) |
| 2 | Fused \(a^2\) | mul | 1 | \(n\) |
| **Fused total** | | | | **\(2n\)** |

Forward FLOPs are **identical** for identity and fused paths; fusion elides HBM traffic for \(\mathrm{ReLU}(X)\), not forward arithmetic.

### 4.3 Paper-comparable (if different)

Primer defines the activation as \(\mathrm{ReLU}(x)^2\) with the same two primitive ops (ReLU + square). **No cheaper paper-comparable forward formula** — Appendix E style MLP accounting that omits activations entirely is an omission, not a lower ReLU² constant.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{ReLU}^2,\mathrm{fwd}} = 2 \cdot n = 2 \cdot S \cdot d.
\]

**Arithmetic intensity (numeric example):** Nemotron-style prefill, \(S{=}8192\), \(d_{\mathrm{ff}}{=}14336\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S \cdot d_{\mathrm{ff}}\) | \(\approx 1.18 \times 10^8\) |
| FLOPs | \(2n \approx 2.35 \times 10^8\) |
| Min HBM bytes (read \(X\) + write \(Y\), fused) | \(2 n e \approx 452\) MiB |
| Arithmetic intensity | \(\approx 0.12\) FLOP/byte — **memory-bound** |

At \(d_{\mathrm{ff}}{=}21504\): \(2n \approx 3.5 \times 10^8\) FLOPs/layer — ~**0.6%** of Up+Down GEMM FLOPs (\(\approx 57\)B/layer at \(S{=}8192\)), roughly half the SiLU gate share (\(\approx 1.5%\)) because ReLU² is cheaper per element (\(2\) vs \(5\) FLOPs) but lacks a gated multiply partner.

---

## Section 5: Backward FLOPs — step-by-step derivation

Let \(A = \mathrm{ReLU}(X)\), \(Y = A^2\).

\[
\frac{\partial Y}{\partial X} = \begin{cases} 2x & x > 0 \\ 0 & x \le 0 \end{cases} = 2 \cdot \mathrm{ReLU}(x).
\]

\[
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \cdot 2 \cdot \mathrm{ReLU}(X).
\]

PyTorch autograd through the eager chain saves **input \(X\)** for ReLU and **\(\mathrm{ReLU}(X)\)** for `square`; a fused leaf can **recompute \(\mathrm{ReLU}(X)\) from saved \(X\)** (SiLU-style policy).

### 5.1 Identity lowering (unfused `Maximum → Multiply`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Multiply` VJP (self-square): `grad_y * A` (left) | mul | 1 | \(n\) |
| 2 | `Multiply` VJP (self-square): `grad_y * A` (right) | mul | 1 | \(n\) |
| 3 | `Maximum` VJP: `grad_a * (X > 0)` | mul | 1 | \(n\) |
| **Identity backward total** | | | | **\(3n\)** |

Steps 1–2 sum to \(\mathrm{grad}_A = 2 \cdot \mathrm{grad}_Y \cdot A\); step 3 propagates to \(X\). When \(x \le 0\), \(A = 0\) forces zero gradient without an extra comparison FLOP.

### 5.2 Fused region leaf (kernel-accurate, recompute from saved \(X\))

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(\mathrm{ReLU}(x)\) from saved \(x\) | max | 1 | \(n\) |
| 2 | \(2 \cdot \mathrm{ReLU}(x)\) | mul (const 2) | 1 | \(n\) |
| 3 | `grad_out * (2 * relu(x))` | mul | 1 | \(n\) |
| **Fused backward total** | | | | **\(3n\)** |

`requires_grad=False` → backward FLOPs = **0**.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{ReLU}^2,\mathrm{bwd}} = \begin{cases} 3 \cdot n & \text{if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

**Common mistake:** Do **not** use \(2 \times 2 = 4\) (GEMM-style “backward ≈ 2× forward”) — the derivative \(2 \cdot \mathrm{ReLU}(x)\) fuses the square and ReLU VJPs into **3** FLOPs/element, not 4.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Identity (unfused)** | \(\mathrm{ReLU}(X)\), \(Y\) | \(2 n e\) | — |
| **Fused `region/squared_relu`** | \(Y\) only | \(n e\) | \(\mathrm{ReLU}(X)\) (\(n e\)) |

Input \(X\) is a boundary edge (not re-allocated by the region).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| PyTorch eager chain | input \(X\); square input \(A=\mathrm{ReLU}(X)\) | \((S,d)\) each | up to \(2 n e\) | autograd per-op |
| **`region/squared_relu` (proposed)** | input \(X\) only | \((S,d)\) | \(n e\) | `save_input: true`; recompute \(\mathrm{ReLU}(X)\) in backward (+1 FLOP/elem vs saving \(A\)) |
| Unfused identity | ReLU saves \(X\); Multiply saves both \(A, A\) | — | up to \(n e + 2 n e\) redundant | chain-dependent |

**Tradeoff:** fused region with `save_input` alone matches SiLU/ReLU saved-input policy (\(n e\) bytes) and avoids retaining a separate \(\mathrm{ReLU}(X)\) buffer (\(n e\) saved).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(relu_out) → ALLOCATE(Y) → SAVE(input) → SAVE(relu_out)
```

**Fused region leaf (`region/squared_relu`):**
```
ALLOCATE(Y) → SAVE(input)
```

Backward phase end: `RELEASE(input)` on the saved input edge.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF `ReLUSquaredActivation` | Transformers | No (2 ops) | standalone | \(2n\) | \(3n\) | Save \(X\) + \(\mathrm{ReLU}(X)\) | — |
| PyTorch ATen | torch | Per-op | standalone | \(2n\) | \(3n\) | Per-op autograd | — |
| Primer | google-research | Framework | MLP | \(2n\) | \(3n\) | Framework | — |
| Zepto identity | `ReLU` + `Multiply` | No | standalone | \(2n\) | \(3n\) | Chain-dependent | — |
| Zepto (proposed) | `region/squared_relu` | Yes (cost leaf) | standalone | \(2n\) | \(3n\) | SAVE(input) | **to_implement** |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: squared_relu
recommended_region_ids:
  - id: region/squared_relu
    variant: default
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: to_implement
    capabilities:
      - squared_relu
      - relu2
    recipe:
      forward_flops_per_element: 2
      backward_flops_per_element: 3
      save_input: true
      save_relu_intermediate: false
      elided_temps:
        - relu_out
      saved_backward:
        - name: input
          shape: "(S, d)"
      resource_events_forward:
        - "ALLOCATE(Y) → SAVE(input)"
      resource_events_backward:
        - "RELEASE(input) on backward phase end"
      numerics_tags:
        - comparison_not_billed
        - zero_subgradient_at_origin
pattern_rule:
  op_families:
    - maximum
    - multiply
  edge_constraints:
    - "maximum.output → multiply.left and multiply.right (self-square of relu output)"
  constraints:
    - right_operand_is_zero
    - squared_relu_mul_relu_relu
recipe:
  forward_flops: "2 * numel(output)"
  backward_flops: "3 * numel(output) if requires_grad else 0"
  forward_flops_per_element: 2
  backward_flops_per_element: 3
  save_input: true
  save_relu_intermediate: false
  elided_temps:
    - relu_out
  saved_backward:
    - name: input
      shape: "(S, d)"
  resource_events_forward:
    - "ALLOCATE(Y) → SAVE(input)"
  resource_events_backward:
    - "RELEASE(input) on backward phase end"
  numerics_tags:
    - comparison_not_billed
    - zero_subgradient_at_origin
capabilities:
  - squared_relu
  - relu2
priority: 10
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- So et al. (2021), Primer — squaring ReLU activations: https://arxiv.org/abs/2109.08668
- HuggingFace `ReLUSquaredActivation` / `relu2`: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Nemotron config (`hidden_act: relu2`): https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron/configuration_nemotron.py
- PyTorch ReLU (`Activation.cpp`): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- PyTorch `torch.square`: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorFactories.cpp
- Primer reference implementation: https://github.com/google-research/google-research/tree/master/primer
- Zepto ReLU module: `src/zepto/modules/relu.py`
- Zepto ReLU region (pattern reference): `src/zepto/analysis/lowering/implementations/regions/relu/`
- Zepto SiLU region (composed-activation pattern reference): `src/zepto/analysis/lowering/implementations/regions/silu/`
- Model architecture gaps (ReLU² MLP): `docs/model-architecture-gaps-2026-09-14.md`

**Verification date:** 2026-09-14

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P1 | `region/squared_relu` | §8 | standalone_activation | Unblocks Primer/Nemotron/Persimmon/BitNet MLP costing; elides `relu_out` HBM temp; no Liger/Hub fused alternative today |
| P2 | `ReLUSquared` module | §0 | — | Reusable module + provenance rule (`prov-squared-relu`) for pattern discovery |
| P2 | Two-projection MLP module | §3 | MLP stack | Compose `Linear → region/squared_relu → Linear` per `docs/model-architecture-gaps-2026-09-14.md` |

**Open gaps:**
- **No PyTorch `aten::relu_squared`** — costing assumes sequential fused per-op kernels; a future dedicated op would not change the \(2n/3n\) arithmetic leaf.
- **No Liger / Hub kernel** — unlike SiLU/GELU, ReLU² has no third-party fused training kernel to model as a separate variant.
- **Mamba/Nemotron-H hybrid blocks** — ReLU² costing is leaf-only; Mamba-2, MoE, and single-mixer schedulers are out of scope (see model-architecture gaps doc).
