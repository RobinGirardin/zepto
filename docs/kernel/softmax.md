# Zepto kernel research: aten::softmax

**Date:** 2026-09-07
**Scope:** Standalone row softmax (fusion boundary **A**); PyTorch ATen `aten::softmax` as reference fused kernel
**Context:** Prefill and decode; training and inference; CUDA primary, ROCm (AITER) secondary; kernel-accurate Zepto leaf

---

## Section 0: Mathematical definition

For input logits \(z \in \mathbb{R}^{(\ldots, S)}\) with softmax along the **last** axis (keys, length \(S\)):

\[
m_i = \max_j z_{ij}, \qquad
p_{ij} = \frac{e^{z_{ij} - m_i}}{\sum_k e^{z_{ik} - m_i}}.
\]

**I/O shapes (attention tile):** input \(z\) and output \(P\) both \((h, S, S)\) or batched \((B, h, S, S)\). Reduction is over the last dimension.

**Numerics policies (bytes, not FLOPs):**
- **Stable max-subtract:** ATen/HF always subtract row max before `exp` ([`SoftMax.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu)).
- **HF fp32 softmax:** `softmax(..., dtype=torch.float32)` doubles working-set bytes during the kernel; **0 extra FLOPs** ([HF Llama eager](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)).
- **Structural causal mask** (`is_causal=True` in SDPA/Flash) is **not** part of `aten::softmax`; it lives in boundary C (§3 cross-ref).

**Zepto reference module:** `src/zepto/modules/softmax.py` decomposes `scale → exp → reduce_sum → divide` and **omits** stable max-subtract — structural decomposition only, not numerically identical to ATen.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch `F.softmax` → **`aten::softmax`** (CUDA `SoftMax.cu`; cuDNN optional/legacy); ROCm AITER `softmax` ([kernels-community/aiter-kernels](https://huggingface.co/kernels-community/aiter-kernels)) |
| **Identity lowering** | Unfused semantic primitives | Zepto `Softmax`: optional `Multiply(scale)` → `Exp` → `ReduceSum` → `Divide`; each intermediate is full-rank \((h,S,S)\) `ALLOCATE` |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/softmax` — stable softmax **5 FLOPs/element**, elides `exp_scores` and on-chip max/subtract temps; writes \(P\) only |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **PyTorch ATen** | `torch.nn.functional.softmax` → `aten::softmax` | Yes (single dispatch) | A | CUDA default; CPU/MPS/XPU separate | Both |
| **PyTorch eager (manual)** | `exp → sum → div` or Zepto primitives | No | A | Any | Both |
| **HF Transformers eager** | `matmul → add(mask) → softmax(fp32)` | Softmax step fused via ATen | A (after separate mask add) | CUDA | Both |
| **Megatron / TE** | `scaled_masked_softmax_cuda` | Yes (scale+mask+softmax) | **B** | CUDA | Training |
| **AITER softmax** | Hub `aiter-kernels` | Yes | A | ROCm | Both |
| **FlashAttention / SDPA-flash** | Online softmax inside attention | Yes (with QKᵀ+PV) | **C** | CUDA/ROCm/MPS | Both |
| **Zepto (today)** | `Softmax` module in GQA | No | A unfused | Any | Both |
| **Zepto (planned)** | `region/softmax` | Yes (cost leaf) | A | `any` + optional `cuda_only` | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone row softmax on a materialized tile; maps to `region/softmax`.

**Mutually exclusive:**
- `region/softmax` (A) vs `region/masked_softmax` (B) on the **same** logits tile when scale+mask are separate ops — pick B if fusing scale+mask+softmax; pick A for bare `aten::softmax` on already-masked/scaled logits.
- `region/softmax` / `region/masked_softmax` vs `region/gqa/*` (C) on the same attention layer — C replaces score + softmax + context entirely.

**Composable:**
- `region/softmax` composes with upstream `MatMul(Q,K^T)` and downstream `MatMul(P,V)` when not using boundary C.
- Liger patches (RMSNorm, CE, SwiGLU) compose with FlashAttention Hub kernels; they do not replace standalone softmax.

**Execution constraints:**
- HBM traffic for `aten::softmax` is **not** guaranteed 1-pass: row must fit SRAM (\(S{=}8192\) bf16 → 16 KiB; \(S{=}65536\) → 128 KiB may spill).
- Do not bill `aten::softmax` as 1 HBM read + 1 write at 64k context without noting SRAM assumption.

---

## Section 4: Forward FLOPs — step-by-step derivation

Tile shape: \((h, S, S)\), \(|tile| = h S^2\) elements.

### 4.1 Identity lowering (Zepto `Softmax` module, no stable max)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Scale (optional) | `Multiply` by \(1/\sqrt{d_h}\) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 2. Exp | `Exp` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 3. Sum | `ReduceSum` over keys | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 4. Normalize | `Divide` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| **Unfused (with scale)** | | | | **\(4 h S^2\)** |

This chain **omits** row-max and subtract; it is not kernel-accurate for `aten::softmax`.

### 4.2 Fused region leaf (kernel-accurate, stable softmax)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Max | `ReduceMax` / comparisons | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 2. Subtract | \(z_{ij} - m_i\) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 3. Exp | `Exp` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 4. Sum | `ReduceSum` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 5. Divide | normalize | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| **Stable fused total** | | | | **\(5 h S^2\)** |

### 4.3 Paper-comparable (if different)

Apertus Appendix E: \(\mathrm{FLOPs}_{\mathrm{softmax}} = 3 h S^2\) (exp + sum + div only, **no max-subtract**). Label: **paper-comparable only** — not the Zepto fused-region leaf.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 5\, h S^2
\]

**Arithmetic intensity (numeric example):** Apertus-8B GQA prefill, \(h{=}32\), \(S{=}8192\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| FLOPs | \(5 \times 32 \times 8192^2 \approx 1.07 \times 10^{10}\) |
| Ideal 1-pass HBM (read \(z\), write \(P\)) | \(4 h S^2 e = 8.0\) GiB |
| AI | \(5/4 = 1.25\) FLOP/byte |

Unfused eager with separate `exp`/`sum` buffers: multiple HBM passes → effective AI **drops** below 1.25.

---

## Section 5: Backward FLOPs — step-by-step derivation

VJP with saved output \(P\): \(dZ = P \odot (dY - \langle P, dY \rangle)\).

| Step | Primitive | Per element | FLOPs |
|------|-----------|-------------|-------|
| Row dot | \(\langle P, dY \rangle\) | reduction billing | \(h S^2\) |
| Subtract | \(dY - \text{dot}\) | 1 sub | \(h S^2\) |
| Multiply | \(P \odot \ldots\) | 1 mul | \(h S^2\) |
| Extra chain ops | chain rule bookkeeping | ~1 op/elem | \(h S^2\) |
| **Total** | | | **\(\approx 4 h S^2\)** |

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} \approx 4\, h S^2
\]

(`requires_grad=False` → backward FLOPs = 0.)

If pre-softmax logits \(z\) are saved instead of \(P\), order is the same. FlashAttention (boundary C) recomputes tiles — not applicable to standalone `region/softmax`.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

Worked unit: one \((h,S,S)\) bf16 buffer at \(h{=}32\), \(S{=}8192\) = **4.0 GiB**.

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Unfused Zepto Softmax** | `scores`, `exp_scores`, `weights` (\(P\)) | **8–12 GiB** (2–3× tile) | — |
| **HF eager + fp32 softmax** | + fp32 working set | **8.0 GiB** tile bytes; peak up to ~16 GiB if bf16 temps overlap | — |
| **Fused `region/softmax` (ATen-style)** | **`P` only** | **4.0 GiB** | `exp_scores`, on-chip max/subtract |
| **AITER softmax (ROCm)** | **`P` only** | same as fused A | same |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Eager / fused `region/softmax` | Output **\(P\)** | \((h,S,S)\) | \(\Theta(h S^2)\) | `save_P: true` |
| Save logits \(z\) instead | Pre-softmax tile | \((h,S,S)\) | \(\Theta(h S^2)\) | alternative recipe variant |

### 6.3 Resource event chains

**Identity lowering (unfused Zepto Softmax with scale):**
```
ALLOCATE(scores) → ALLOCATE(exp_scores) → ALLOCATE(P) → SAVE(P)
```

**Fused region leaf (`region/softmax`, boundary A):**
```
ALLOCATE(P) → SAVE(P)
```

On-chip temps (max, subtract, exp partials) are **elided** — not `ALLOCATE`/`SAVE` in the resource stream.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| ATen CUDA | PyTorch | Yes | A | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/softmax/cuda` |
| ATen reference | PyTorch (any HW) | Yes (cost leaf) | A | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/softmax/reference` |
| AITER | aiter-kernels | Yes | A | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/softmax/rocm` (future) |
| Zepto unfused | `modules/softmax.py` | No | A | \(4 h S^2\) (no max) | ~\(3 h S^2\) | 2–3× tile peak | identity |
| Megatron TE | TransformerEngine | Yes | B | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/masked_softmax` |
| FlashAttention | flash-attn | Yes | C | §11 | §11 | No \((h,S,S)\) | `region/gqa/*` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: softmax
status: to_implement
recommended_region_ids:
  - impl_id: region/softmax/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - impl_id: region/softmax/cuda
    variant: cuda
    hardware_gate: cuda_only
    fusion_boundary: A
    status: to_implement
pattern_rule:
  op_families:
    - multiply   # optional scale from Softmax(scale=...)
    - exp
    - reduce_sum
    - divide
recipe:
  forward_flops: "5 * numel(output)"
  backward_flops: "4 * numel(output) if requires_grad else 0"
  forward_flops_per_element: 5
  backward_flops_per_element: 4
  save_P: true
  elided_temps:
    - exp_scores
    - row_max        # on-chip only
    - shifted_logits # on-chip only
  saved_backward:
    - name: P
      shape: "(h, S, S) or batched"
  resource_events_forward:
    - "ALLOCATE(P) → SAVE(P)"
  resource_events_backward:
    - "RELEASE(P) on backward phase end"
  numerics_tags:
    - stable_max_subtract
capabilities:
  - fused
priority: 5
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- PyTorch ATen CUDA SoftMax: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu
- Triton fused softmax tutorial (HBM traffic baseline): https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- HF Llama eager attention (fp32 softmax policy): https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- FlashAttention paper (online softmax, boundary C contrast): https://arxiv.org/abs/2205.14135
- AITER ROCm softmax: https://huggingface.co/kernels-community/aiter-kernels
- Megatron fused softmax (boundary B contrast): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py
- Zepto domain doc §10: `docs/kernel-implementation.md`

**Verification date:** 2026-09-07

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P1 | `region/softmax/reference` | §8 | A | Model `aten::softmax` kernel-accurate leaf; fix G4b false peak from unfused `exp_scores` temps in GQA |
| P2 | `region/softmax/cuda` | §8 | A | Hardware-gated variant matching CUDA ATen dispatch |
| P3 | `region/masked_softmax` | docs §10 | B | Separate backlog — scale+mask+softmax for Megatron/TE path |
| — | G4b | §6.1 | — | Unfused GQA sums scores+exp+weights → 8–12 GiB false peak per layer |

**Open gaps:**
- **G4b:** Identity GQA path inflates peak VRAM; `region/softmax` fixes softmax sub-chain only — full GQA fusion is `region/gqa/*`.
- **Module vs ATen numerics:** `Softmax` module omits stable max-subtract; fused region bills 5×/elem (ATen-accurate FLOPs) while pattern matches 3–4 op decomposed chain — document mismatch in architecture.
- **ROCm variant:** AITER `softmax` deferred (`region/softmax/rocm`) until hardware_gate routing is needed.
- **HBM pass count:** Do not assume 1-pass at \(S > 32k\); SRAM spill is implementation-dependent, not modeled in FLOP leaf.
