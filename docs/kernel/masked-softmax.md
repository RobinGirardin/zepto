# Zepto kernel research: Megatron/TE masked softmax

**Date:** 2026-09-07
**Scope:** Fusion boundary **B** — fused scale + additive mask + stable row softmax on materialized \((h,S,S)\) logits; Megatron-LM / Apex and NVIDIA TransformerEngine training kernels
**Context:** GQA prefill, bf16 primary, causal additive mask; training and inference; CUDA; kernel-accurate Zepto leaf (not full FlashAttention boundary C)

---

## Section 0: Mathematical definition

Attention score logits for query row \(i\), key column \(j\) (heads \(h\), sequence \(S\), head dim \(d_h\)):

\[
z_{ij} = \frac{(QK^\top)_{ij}}{\sqrt{d_h}} + M_{ij}, \qquad
m_i = \max_j z_{ij}, \qquad
p_{ij} = \frac{e^{z_{ij}-m_i}}{\sum_k e^{z_{ik}-m_i}}.
\]

**I/O shapes:** input raw scores \((QK^\top)\) or pre-mask logits; additive mask \(M\) broadcastable to \((B,h,S,S)\) or \((1,1,S,S)\); output weights \(P\) same shape as logits tile \((h,S,S)\) or batched \((B,h,S,S)\). Softmax reduces along the **last** axis (keys).

**Megatron/TE fused kernel** consumes unscaled \(QK^\top\), applies \(1/\sqrt{d_h}\), adds \(M\), and runs numerically stable softmax in one CUDA launch — no materialized intermediate \((h,S,S)\) buffers for scaled/masked logits or `exp_scores`.

**Numerics policies (bytes, not FLOPs):**
- Causal training uses upper-triangular masked variant (`scaled_upper_triang_masked_softmax_cuda`) or general mask tensor.
- HF eager uses `softmax(..., dtype=torch.float32)` — **0 extra FLOPs**, **2×** softmax tile bytes while the kernel runs.
- Structural causal (`is_causal=True` in SDPA/Flash) is **boundary C**, not this leaf — **0 mask bytes** there.

**Zepto identity reference:** `GroupedQueryAttention` → `Add(causal_mask)` → `Softmax(scale=1/√d_h)` ([`gqa.py`](../src/zepto/modules/gqa.py), [`softmax.py`](../src/zepto/modules/softmax.py)).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production NVIDIA training baseline | Megatron-LM `scaled_masked_softmax_cuda`, `scaled_upper_triang_masked_softmax_cuda` ([`fused_softmax.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py)); TE `te_softmax::scaled_masked_softmax_fwd` ([TE softmax](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/softmax.py)) |
| **Identity lowering** | Unfused semantic primitives in Zepto GQA | `Add(mask)` → `Multiply(scale)` → `Exp` → `ReduceSum` → `Divide`; each intermediate full-rank \((h,S,S)\) `ALLOCATE` |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/masked_softmax` — **\(5 h S^2\)** FLOPs (scale+mask+exp+sum+div billing); writes **\(P\) only**; elides pre-softmax logits |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Megatron-LM / Apex** | `scaled_masked_softmax_cuda`, `scaled_upper_triang_masked_softmax_cuda`, `ScaledSoftmax`, `SoftmaxOne` | Yes (scale+mask+softmax) | **B** | CUDA; shape/seq-len fallbacks to `torch.softmax` | Training primary |
| **NVIDIA TransformerEngine** | `te_softmax::scaled_masked_softmax_fwd` / `_bwd` | Yes | **B** | CUDA | Training |
| **HF Transformers eager** | `matmul → add(causal_mask) → softmax(fp32)` | Softmax via ATen; mask separate | A (after mask add) | CUDA | Both |
| **PyTorch ATen `F.softmax`** | Standalone on already-masked logits | Yes (softmax only) | A | CUDA | Both |
| **FlashAttention / SDPA-flash** | Online softmax inside attention | Yes (with QKᵀ+PV) | **C** | CUDA/ROCm/MPS | Both |
| **Zepto (today)** | GQA decomposed `Add → Softmax` | No | A unfused | Any | Both |
| **Zepto (registered)** | `region/softmax` | Yes (softmax only, boundary A) | A | any / cuda_only | Both |
| **Zepto (planned)** | `region/masked_softmax` | Yes (boundary B leaf) | **B** | cuda_only (Megatron/TE) | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **B** — fused **scale + additive mask + stable softmax** on a materialized \((h,S,S)\) tile; maps to `region/masked_softmax`.

**Mutually exclusive:**
- `region/masked_softmax` (B) vs `region/gqa/*` / Flash / Metal-Flash / Sage (C) on the **same attention layer** — C replaces \(QK^\top\) + score softmax + \(PV\) entirely.
- `region/masked_softmax` (B) vs `region/softmax` (A) on the **same logits tile** when scale+mask are fused — pick B for Megatron/TE; pick A for bare `aten::softmax` on pre-masked logits.

**Composable:**
- `region/masked_softmax` composes with upstream `MatMul(Q,K^T)` and downstream `MatMul(P,V)` when not using boundary C.
- Liger patches (RMSNorm, CE, SwiGLU) compose with FlashAttention Hub kernels; they do **not** provide a boundary-B masked-softmax leaf.
- Boundary **D** (`region/linear_ce`) is orthogonal — vocab-axis CE, not key-axis causal masking.

**Execution constraints:**
- Megatron kernels require CUDA, specific head/seq divisibility; fall back to decomposed PyTorch softmax on unsupported shapes.
- Causal upper-triangular variant avoids storing dense \((S,S)\) mask when applicable — mask bytes may be **0** for structural upper-tri, but FLOP leaf remains **\(5 h S^2\)** on the logits tile.
- Do not bill guaranteed 1-pass HBM at \(S > 32k\); SRAM spill is implementation-dependent.

---

## Section 4: Forward FLOPs — step-by-step derivation

Tile shape: \((h, S, S)\), \(|tile| = h S^2\) elements. \(QK^\top\) GEMM (\(2 h S^2 d_h\)) is **outside** this leaf (§11 GQA).

### 4.1 Identity lowering (Zepto GQA score path after \(QK^\top\))

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Scale | `Multiply` by \(1/\sqrt{d_h}\) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 2. Mask | `Add` materialized mask | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 3. Exp | `Exp` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 4. Sum | `ReduceSum` over keys | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 5. Normalize | `Divide` | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| **Unfused total** | | | | **\(5 h S^2\)** |

This identity chain **omits** row-max and subtract (matches Zepto `Softmax` module, not ATen stable softmax). Megatron/TE kernels execute max+subtract **on-chip** without extra `ALLOCATE`.

### 4.2 Fused region leaf (kernel-accurate, Megatron/TE boundary B)

Megatron/TE fuse scale + mask + stable softmax. Zepto bills the **same five elementwise stages** as the unfused score path above — they run inside one kernel, not as free ops:

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Scale | multiply by \(1/\sqrt{d_h}\) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 2. Mask | add \(M_{ij}\) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 3. Exp | stable exp after on-chip max-sub | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 4. Sum | row sum | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 5. Divide | normalize | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| **Fused B total** | | | | **\(5 h S^2\)** |

On-chip max + subtract add **\(2 h S^2\)** hardware work in stable kernels but **do not** increment the Zepto leaf beyond \(5 h S^2\) and **do not** allocate extra tensors.

### 4.3 Paper-comparable (if different)

Apertus Appendix E: \(\mathrm{FLOPs}_{\mathrm{softmax}} = 3 h S^2\) (exp + sum + div only, **no scale/mask/max-subtract**). Label: **paper-comparable only** — not the Zepto fused-region leaf.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 5\, h S^2
\]

**Arithmetic intensity (numeric example):** Apertus-8B GQA prefill, \(h{=}32\), \(S{=}8192\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| FLOPs | \(5 \times 32 \times 8192^2 \approx 1.07 \times 10^{10}\) |
| Ideal 1-pass HBM (read scores+mask, write \(P\)) | \(\approx 4 h S^2 e = 8.0\) GiB (mask often broadcast / structural) |
| AI (1-pass bound) | \(5/4 = 1.25\) FLOP/byte on logits tile |

---

## Section 5: Backward FLOPs — step-by-step derivation

VJP with saved output \(P\): \(dZ = P \odot (dY - \langle P, dY \rangle)\).

| Step | Primitive | Per element | FLOPs |
|------|-----------|-------------|-------|
| Row dot | \(\langle P, dY \rangle\) | reduction | \(h S^2\) |
| Subtract | \(dY - \text{dot}\) | 1 sub | \(h S^2\) |
| Multiply | \(P \odot \ldots\) | 1 mul | \(h S^2\) |
| Chain bookkeeping | | ~1 op/elem | \(h S^2\) |
| **Total** | | | **\(\approx 4 h S^2\)** |

Megatron/TE provide fused backward (`scaled_masked_softmax_bwd`, TE `_bwd`); Zepto bills the same **\(4 h S^2\)** leaf as boundary A softmax.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} \approx 4\, h S^2
\]

(`requires_grad=False` → backward FLOPs = 0.)

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

Worked unit: one \((h,S,S)\) bf16 buffer at \(h{=}32\), \(S{=}8192\) = **4.0 GiB**.

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Unfused Zepto GQA score path** | `scores`, `masked_scores`, `exp_scores`, `weights` (\(P\)) | **8–12 GiB** (2–3× tile) | — |
| **HF eager + fp32 softmax** | + fp32 working set | **8.0 GiB** tile bytes; peak up to ~16 GiB if bf16 temps overlap | — |
| **Fused `region/masked_softmax` (Megatron/TE)** | **`P` only** | **4.0 GiB** | `scores`, `scaled_scores`, `exp_scores`, on-chip max/sub |
| **FlashAttention (boundary C)** | **none** \((h,S,S)\) | context \((h,S,d_h)\) only | entire score tile |

Persistent causal mask \(M\): eager HF may store \((1,S,S)\) or \((S,S)\); Megatron upper-tri variant may bill **0** mask bytes when structural.

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Megatron/TE fused masked softmax | Output **\(P\)** | \((h,S,S)\) | \(\Theta(h S^2)\) | `save_P: true` |
| HF eager (typical) | **\(P\)** after fp32 softmax | \((h,S,S)\) | \(\Theta(h S^2)\) | `save_P: true` |
| FlashAttention (C) | Row stats \((m,\ell)\) per query | \(\Theta(h S)\) | §11 cross-ref | not applicable to B |

### 6.3 Resource event chains

**Identity lowering (unfused GQA score path: Add → Softmax decomposed):**
```
ALLOCATE(scores) → ALLOCATE(masked_scores) → ALLOCATE(exp_scores) → ALLOCATE(P) → SAVE(P)
```

**Fused region leaf (`region/masked_softmax`, boundary B):**
```
ALLOCATE(P) → SAVE(P)
```

On-chip temps (scale, mask add, max, subtract, exp partials) are **elided** — not `ALLOCATE`/`SAVE` in the resource stream. Upstream \(QK^\top\) output may be released before the fused kernel when scores are consumed in-register only (Megatron path); Zepto region boundary is **mask-add through softmax**, not the GEMM.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Megatron-LM | megatron.core.fusions | Yes | B | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/masked_softmax/megatron` |
| TransformerEngine | transformer_engine | Yes | B | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/masked_softmax/te` |
| HF eager GQA | transformers | No | A unfused | \(5 h S^2\) score path | \(4 h S^2\) | 2–3× tile peak | identity |
| ATen softmax | PyTorch | Yes (softmax only) | A | \(5 h S^2\) | \(4 h S^2\) | Write \(P\) only | `region/softmax/*` |
| FlashAttention | flash-attn | Yes | C | §11 | §11 | No \((h,S,S)\) | `region/gqa/*` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: masked_softmax
status: to_implement
recommended_region_ids:
  - impl_id: region/masked_softmax/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: B
    status: to_implement
  - impl_id: region/masked_softmax/megatron
    variant: megatron
    hardware_gate: cuda_only
    fusion_boundary: B
    status: to_implement
  - impl_id: region/masked_softmax/te
    variant: te
    hardware_gate: cuda_only
    fusion_boundary: B
    status: to_implement
pattern_rule:
  op_families:
    - add          # causal / padding mask
    - multiply     # 1/sqrt(d_h) scale (from Softmax submodule)
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
    - scores
    - scaled_scores
    - masked_scores
    - exp_scores
    - row_max
    - shifted_logits
  saved_backward:
    - name: P
      shape: "(h, S, S) or batched (B, h, S, S)"
  resource_events_forward:
    - "ALLOCATE(P) → SAVE(P)"
  resource_events_backward:
    - "RELEASE(P) on backward phase end"
  numerics_tags:
    - stable_max_subtract
    - additive_mask
    - megatron_te_fused
capabilities:
  - fused
  - masked_softmax
priority: 6
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Megatron-LM fused softmax: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py
- TransformerEngine scaled masked softmax: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/softmax.py
- HF Llama eager attention (mask add + fp32 softmax): https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- FlashAttention paper (boundary C contrast): https://arxiv.org/abs/2205.14135
- Zepto domain doc §10: `docs/kernel-implementation.md`
- Zepto GQA module: `src/zepto/modules/gqa.py`

**Verification date:** 2026-09-07

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P1 | `region/masked_softmax/reference` | §8 | B | Cost leaf for fused scale+mask+softmax; fixes G4b false peak on GQA score sub-chain |
| P2 | `region/masked_softmax/megatron` | §8 | B | Route when backend profile selects Megatron training stack |
| P3 | `region/masked_softmax/te` | §8 | B | Route when backend profile selects TransformerEngine |
| — | G4b | §6.1 | — | Unfused GQA sums scores+masked+exp+weights → 8–12 GiB false peak per layer |

**Open gaps:**
- **G4b:** Identity GQA path inflates peak VRAM; `region/masked_softmax` fixes score sub-chain only — full elision requires `region/gqa/*` (boundary C).
- **Discovery wiring:** No standalone `MaskedSoftmax` module; region must match `Add → Softmax` subgraph inside `GroupedQueryAttention` via pattern rule (5-op chain) — architecture must define edge constraints.
- **Mask storage:** Upper-triangular Megatron variant vs dense \((S,S)\) mask — byte accounting differs; FLOP leaf unchanged.
- **Stable vs identity FLOPs:** Zepto identity `Softmax` omits max-subtract (4 ops inside Softmax + 1 Add = 5 total); fused B bills 5×/elem including on-chip stable stages — document in architecture.
- **Mutual exclusion with `region/softmax`:** When only softmax is fused (mask already applied), boundary A applies; pattern overlap must be resolved by op-count / provenance priority.
