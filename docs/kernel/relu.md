# Zepto kernel research: ReLU (`saved_input` variant)

**Date:** 2026-09-08
**Proposer:** Robin Girardin
**Scope:** Standalone elementwise activation leaf; `saved_input` backward-retention variant; prefill + decode; training + inference; CUDA primary
**Context:** PyTorch autograd often retains full input \(X\) for ReLU backward instead of a compressed bool mask; Zepto models this as `region/relu/saved_input`

---

## Section 0: Mathematical definition

Rectified Linear Unit (ReLU), standard in modern MLPs ([Nair & Hinton, 2010](https://www.cs.toronto.edu/~hinton/absps/relu.pdf)):

\[
\mathrm{ReLU}(x) = \max(x, 0) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}
\]

**I/O shapes:** input \(X \in \mathbb{R}^{S \times d}\) (Zepto rank-2); output \(Y\) same shape. Elementwise — no reduction axes.

**Numerics policies:**
- Comparison \(x > 0\) is **not** billed as an arithmetic FLOP (Zepto / Atto ReLU policy).
- `max(x, 0)` bills **1 FLOP per element** (single `Maximum` primitive).
- Subgradient at \(x = 0\) is conventionally 0 (PyTorch `aten::relu`).

**Variant focus:** `saved_input` retains the **full input tensor** \(X\) for backward (PyTorch `ThresholdBackward` convention) instead of a separate bool `relu_mask`. FLOPs are unchanged; VRAM for saved activations increases from \(n\) to \(n \cdot e\) bytes.

**Zepto identity reference:** `src/zepto/modules/relu.py` — single `Maximum(x, constant_zero)` op.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch `aten::relu` ([Activation.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp)) — autograd saves input \(X\) |
| **Identity lowering** | Unfused semantic primitive | Zepto `Maximum(left, 0)` or `maximum/relu-mask` (bool mask SAVE) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/relu/saved_input` — **\(1 \lvert Y \rvert\)** forward, **\(1 \lvert Y \rvert\)** backward; **SAVE(input)** for training |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Saved backward | Device routing | Training vs inference |
|----------------|------------------|--------|----------------|----------------|----------------------|
| **PyTorch ATen** | `torch.nn.functional.relu` | Yes | Input \(X\) (default autograd) | CUDA / CPU / MPS | Both |
| **Zepto (default region)** | `region/relu` | Yes | Bool `relu_mask` (\(n\) bytes) | Any | Both |
| **Zepto (saved_input)** | `region/relu/saved_input` | Yes | Full input \(X\) (\(n \cdot e\) bytes) | Any | Both |
| **Zepto (op impl)** | `maximum/relu-mask` | Yes | Bool mask | Any | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** Standalone **elementwise activation leaf** — not an attention fusion boundary (A/B/C/D).

**Mutually exclusive:**
- `region/relu` (mask) vs `region/relu/saved_input` on the same region — selected via `requested_capabilities` (`mask_retention` vs `saved_input`).
- Both vs identity `maximum/identity` — region wins at priority 10 when pattern matches.

**Composable:**
- Runs sequentially inside MLP stacks; does not overlap attention-family regions.
- Default discovery uses mask variant when no capability is requested.

**Execution constraints:**
- Right operand must be provably zero (`PatternConstraint(kind="right_operand_is_zero")`).
- `saved_input` variant requires `requested_capabilities={'saved_input'}`.
- Inference with `requires_grad=False`: backward FLOPs = 0; no SAVE events when grad disabled.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert Y \rvert = S \cdot d\). Comparisons: **0 FLOPs**.

### 4.1 Identity lowering (single `Maximum`)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | `Maximum(left, 0)` | max | 1 | \(n\) |
| **Identity total** | | | | **\(n\)** |

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused `max(x, 0)` | max | 1 | \(n\) |
| **Fused total** | | | | **\(n\)** |

Forward FLOPs are **identical** for mask and saved_input variants.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{ReLU,fwd}} = n = S \cdot d
\]

**Arithmetic intensity (numeric example):** Apertus-8B prefill, \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S \cdot d_{\mathrm{ff}}\) | \(\approx 1.76 \times 10^8\) |
| FLOPs | \(n \approx 1.76 \times 10^8\) |
| Min HBM bytes (read \(X\) + write \(Y\)) | \(2 n e \approx 704\) MiB |
| Arithmetic intensity | \(\approx 0.25\) FLOP/byte — **memory-bound** |

---

## Section 5: Backward FLOPs — step-by-step derivation

\[
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \odot \mathbb{1}[X > 0]
\]

With **saved input**, the mask is derived from \(X > 0\) at backward time (same arithmetic cost as mask multiply):

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Mask multiply (`grad_out * (input > 0)`) | mul | 1 | \(n\) |
| **Backward total** | | | | **\(n\)** if `requires_grad` |

`requires_grad=False` → backward FLOPs = **0**.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{ReLU,bwd}} = \begin{cases} n & \text{if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Fused `region/relu/saved_input`** | output \(Y\) | \(n \cdot e\) | — (input is boundary, not re-allocated) |

No auxiliary bool mask buffer — saved activation is the existing input edge.

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| PyTorch autograd (default) | input \(X\) | \((S,d)\) | \(n \cdot e\) | autograd `save_for_backward` |
| **`region/relu/saved_input`** | **input** \(X\) | \((S,d)\) | \(n \cdot e\) | `save_input: true` |
| **`region/relu` (default)** | **relu_mask** (bool) | \((S,d)\) | \(n\) | `save_relu_mask: true` |

**Tradeoff:** saved_input uses **\(e\times\)** more HBM for saved activations (bf16: 2 bytes/elem vs 1 byte bool) but matches PyTorch eager autograd retention and avoids a separate mask ALLOCATE.

### 6.3 Resource event chains

**Fused region leaf (`region/relu/saved_input`):**
```
ALLOCATE(Y) → SAVE(input)
```

Backward phase end: `RELEASE(input)` on the saved input edge.

**Default mask variant (cross-ref):**
```
ALLOCATE(Y) → ALLOCATE(relu_mask) → SAVE(relu_mask)
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|---------------|---------------|-----------------|--------------|
| PyTorch ATen | torch | \(n\) | \(n\) | Save input \(X\) | — |
| Zepto mask | `region/relu` | \(n\) | \(n\) | Bool mask SAVE | registered |
| Zepto saved_input | `region/relu/saved_input` | \(n\) | \(n\) | SAVE(input) | registered |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: relu
status: registered
recommended_region_ids:
  - impl_id: region/relu
    variant: default
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: registered
    capabilities:
      - relu
      - mask_retention
    recipe:
      save_relu_mask: true
      save_input: false
  - impl_id: region/relu/saved_input
    variant: saved_input
    hardware_gate: any
    fusion_boundary: standalone_activation
    status: registered
    capabilities:
      - relu
      - saved_input
    recipe:
      save_relu_mask: false
      save_input: true
pattern_rule:
  op_families:
    - maximum
  constraints:
    - right_operand_is_zero
recipe:
  forward_flops: "1 * numel(output)"
  backward_flops: "1 * numel(output) if requires_grad else 0"
  forward_flops_per_element: 1
  backward_flops_per_element: 1
  save_relu_mask: false
  save_input: true
  elided_temps: []
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
  - relu
  - saved_input
priority: 10
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Nair & Hinton (2010), ReLU: https://www.cs.toronto.edu/~hinton/absps/relu.pdf
- PyTorch `Activation.cpp` (relu + ThresholdBackward): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- HuggingFace MLP activations: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Zepto module: `src/zepto/modules/relu.py`
- Zepto region: `src/zepto/analysis/lowering/implementations/regions/relu/`

**Verification date:** 2026-09-08

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| — | `region/relu/saved_input` | §8 | standalone | **Registered** — PyTorch-style SAVE(input) variant |
| — | `region/relu` | §8 | standalone | **Registered** — bool mask SAVE (default) |

**Open gaps:**
- **Op-level `maximum/relu-saved-input`** not in scope — region-only variant for this pass.
- **Capability routing:** callers must pass `requested_capabilities={'saved_input'}` to select this variant; default remains mask.
- **Generic `Maximum(a, b)`** must remain on `maximum/identity` when right operand is not provably zero.
