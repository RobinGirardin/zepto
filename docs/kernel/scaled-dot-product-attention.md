# Zepto kernel research: Scaled Dot-Product Attention (SDPA)

**Date:** 2026-09-08
**Proposer:** Robin Girardin
**Scope:** Fusion **boundary C** — PyTorch `F.scaled_dot_product_attention` as a **dispatcher** over math / flash / memory-efficient sub-backends for GQA prefill. Training + inference; CUDA/ROCm/MPS primary; kernel-accurate Zepto leaves per pinned sub-backend.
**Context:** Apertus-8B defaults: \(B{=}1\), \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\), \(S{=}8192\), bf16 (\(e{=}2\)). SDPA is **not** one cost model — Zepto requires `attention_backend="sdpa"` plus optional `sdpa_mode` in `state` ([ADR-0009](docs/adr/0009-backend-profiles-and-attention-selection.md)).

**Known Zepto gaps flagged:** **G4b** (unpinned SDPA on MPS falls through to math and materializes scores); **G3** (decode/paged KV is a sibling leaf).

---

## Section 0: Mathematical definition

Vaswani et al. ([2017](https://arxiv.org/abs/1706.03762)) scaled dot-product attention with GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)):

\[
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
\]

**Prefill work tile** (decoder self-attention, one layer):

\[
Q\in\mathbb{R}^{h\times S\times d_h},\quad
K,V\in\mathbb{R}^{h_{\mathrm{kv}}\times S\times d_h},\quad
O\in\mathbb{R}^{h\times S\times d_h}.
\]

PyTorch exposes this as a **single API** — [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — which **dispatches** at runtime to one of several backends ([PyTorch SDPA docs](https://pytorch.org/docs/stable/notes/cuda.html#flash-attention-v2); [Rabe & Staats, 2021](https://arxiv.org/abs/2112.05682) for tiled softmax theory).

**Sub-backend semantics (what is computed — identical across backends):**

Per query position \(i\) and key \(j\):

\[
z_{ij}=\frac{(q_i k_{\pi(i),j}^\top)}{\sqrt{d_h}},\qquad
m_i=\max_j z_{ij},\qquad
p_{ij}=\frac{e^{z_{ij}-m_i}}{\sum_k e^{z_{ik}-m_i}},\qquad
o_i=\sum_j p_{ij}\, v_{\pi(i),j}.
\]

\(\pi(i)=\lfloor i\cdot h_{\mathrm{kv}}/h\rfloor\) maps query heads to KV heads.

**Numerics (bytes, not FLOPs):**
- Scale \(1/\sqrt{d_h}\) folded into GEMM or applied as scalar — **0 extra FLOPs** vs explicit multiply.
- Structural causal mask (`is_causal=True`): **0 mask bytes**, **0 mask-add FLOPs** in flash/mem-efficient tiles.
- `math` backend: may materialize full \((h,S,S)\) scores and \(P\) in HBM; HF-style fp32 softmax tile doubles bytes during softmax.

**Zepto identity reference:** `src/zepto/modules/gqa.py` — `RepeatKV → MatMul(QK^T) → Add(mask) → Softmax(scale→exp→sum→div) → MatMul(PV)`.

**Structural vs eager:** SDPA is one fused API call; Zepto models each **sub-backend** as a distinct `region/gqa` variant with different VRAM elision.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | PyTorch SDPA dispatcher → `math` / `flash` / `mem_efficient` / cuDNN FMHA |
| **Identity lowering** | Structural / debug chain | Zepto `GroupedQueryAttention`: decomposed 10-op chain (§0) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gqa/sdpa-math`, `region/gqa/sdpa-flash`, `region/gqa/sdpa-mem-efficient` |

**Default estimates:** use the **pinned sub-backend leaf** when `attention_backend="sdpa"` and `state.sdpa_mode` is set; unpinned SDPA is **ambiguous** — document as **G4b** risk (worst case = math).

**Two costs — never mixed:**
1. **Theoretical FLOPs** — GEMM + stable online softmax (fusion does not drop GEMM terms).
2. **HBM traffic / peak VRAM** — flash/mem-efficient lower bytes vs math.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| PyTorch SDPA `math` | [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | No (decomposed) | A+GEMM | All | Saves \(P\); fp32 softmax possible |
| PyTorch SDPA `flash` | Same API; FlashAttention backend | Yes | **C** | CUDA/ROCm when available | Recompute backward; \(\Theta(hS)\) stats |
| PyTorch SDPA `mem_efficient` | Same API; xFormers-style tiled | Yes | **C** | CUDA/ROCm | Tiled; no dense \(P\); not identical IO to Flash |
| cuDNN FMHA | SDPA internal on NVIDIA | Yes | **C** | CUDA | Pin explicitly; same asymptotic class as flash |
| HF Transformers eager | [`eager_attention_forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | No | A+GEMM | All | Same memory class as SDPA math |
| FlashAttention-2/3 | [`flash_attn_func`](https://github.com/Dao-AILab/flash-attention) | Yes | **C** | CUDA | Separate `attention_backend=flash_attention_2` leaf |
| Metal-Flash SDPA | Hub [`metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) | Yes | **C** | MPS | MPS analogue of SDPA-flash |
| FlexAttention | PyTorch 2.5+ compiled | Yes | **C** | CUDA | Out of scope until requested |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **C** (full attention: \(QK^\top\) + softmax + \(PV\)) for flash/mem-efficient; **A+GEMM** for math sub-backend (materialized scores).

**Mutually exclusive (same attention layer):**
- `region/gqa/sdpa-*` **replaces** identity chain and `region/masked_softmax` when `attention_backend="sdpa"`.
- `region/gqa/flash2`, `region/gqa/flash3` apply when `attention_backend="flash_attention_2"`, not SDPA.
- `region/gqa/paged` is decode-only (\(S_q{=}1\)) — different work tile.

**Composable:**
- Q/K/V/O linear projections remain separate `region/linear` leaves.
- Liger patches compose around SDPA per [TRL kernels hub](https://huggingface.co/docs/trl/en/kernels_hub).

**Execution constraints:**
- SDPA dispatcher ambiguity: Zepto **must pin** `sdpa_mode` via `state` when `attention_backend="sdpa"`.
- Causal masking structural when `is_causal=True` — no dense \((S,S)\) mask in flash/mem-efficient paths.
- MPS: unpinned SDPA often selects `math` unless Hub metal-flash-sdpa loaded (§11.8 `docs/kernel-implementation.md`).

---

## Section 4: Forward FLOPs — step-by-step derivation

Tile: one layer, prefill, \(B{=}1\), GQA with \(h\) query heads, sequence \(S\), head dim \(d_h\).

### 4.1 Identity lowering (unfused eager)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | RepeatKV (K) | \((h_{\mathrm{kv}},S,d_h)\to(h,S,d_h)\) | 0 | 0 |
| 2 | RepeatKV (V) | same | 0 | 0 |
| 3 | MatMul \(QK^\top\) | \((h,S,d_h)\times(h,d_h,S)\) | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |
| 4 | Add (mask) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 5 | Multiply (scale) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 6 | Exp | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 7 | ReduceSum | per row | \(h S^2\) | \(h S^2\) |
| 8 | Divide | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 9 | MatMul \(PV\) | \((h,S,S)\times(h,S,d_h)\) | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |

Identity subtotal: \(4 h S^2 d_h + 6 h S^2\) (includes mask add).

### 4.2 SDPA `math` region leaf

Same FLOPs as identity — decomposed internally but billed as one leaf:

\[
\mathrm{FLOPs}_{\mathrm{fwd,math}} = 4 h S^2 d_h + 6 h S^2.
\]

### 4.3 SDPA `flash` / `mem_efficient` region leaf (kernel-accurate)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | Block GEMM \(QK^\top\) | SRAM blocks | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |
| 2 | Online softmax | per row in tile | \(5 h S^2\) | \(5 h S^2\) |
| 3 | Block GEMM \(PV\) | SRAM blocks | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |

Causal structural mask: **0** mask-add FLOPs.

\[
\mathrm{FLOPs}_{\mathrm{fwd,flash}} = \mathrm{FLOPs}_{\mathrm{fwd,mem}} = 4 h S^2 d_h + 5 h S^2.
\]

### 4.4 Paper-comparable (Appendix E)

\[
\mathrm{FLOPs}_{\mathrm{fwd,paper}} = 4 h S^2 d_h + 3 h S^2.
\]

**Apertus-8B numeric example** (\(h{=}32\), \(S{=}8192\), \(d_h{=}128\)):
- Flash/mem-efficient: ≈ **1.0845 TFLOPs** per layer prefill
- Math: + \(h S^2\) mask add ≈ **1.0953 TFLOPs**

**Arithmetic intensity (flash leaf, minimum HBM bytes):**
- Inputs/outputs: ≈ **384 MiB** bf16 at \(S{=}8192\)
- AI ≈ **2700** FLOPs/byte (order-of-magnitude)

---

## Section 5: Backward FLOPs — step-by-step derivation

### SDPA `math` (saves \(P\))

Standard softmax + GEMM backward on materialized \(P\):

\[
\mathrm{FLOPs}_{\mathrm{bwd,math}} \approx 8 h S^2 d_h + 10 h S^2.
\]

### SDPA `flash` / `mem_efficient` (recompute)

Same recompute path as FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):

\[
\mathrm{FLOPs}_{\mathrm{bwd,flash}} = \mathrm{FLOPs}_{\mathrm{bwd,mem}} \approx 2 \cdot (4 h S^2 d_h + 5 h S^2) = 8 h S^2 d_h + 10 h S^2.
\]

`requires_grad=False` (inference): backward FLOPs = **0**.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| SDPA math | scores, masked scores, fp32 tile, \(P\) | \(\Theta(h S^2)\) — **4–8 GiB** at \(S{=}8192\) | — |
| SDPA flash / mem_efficient | context output only | \(\Theta(h S d_h)\) | scores, \(P\), fp32 tile, causal mask |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| SDPA math | \(P\) | \((h,S,S)\) | \(\Theta(h S^2)\) | `save_P=true` |
| SDPA flash / mem_efficient | Row stats \((m,\ell)\) | \((h,S)\) fp32 ×2 | \(B h S \cdot 8\) | `save_row_stats=true` |

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(k_repeated) → ALLOCATE(v_repeated) → ALLOCATE(scores) →
ALLOCATE(masked_scores) → ALLOCATE(P) → SAVE(P) → ALLOCATE(context)
```

**SDPA math region leaf:**
```
ALLOCATE(scores) → ALLOCATE(P) → SAVE(P) → ALLOCATE(context)
```

**SDPA flash / mem_efficient region leaf:**
```
ALLOCATE(context) → SAVE(row_stats)   # training only; no (h,S,S) temps
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| SDPA math | PyTorch | No | A+GEMM | \(4 h S^2 d_h + 6 h S^2\) | save \(P\) | \((h,S,S)\) temps | `region/gqa/sdpa-math` |
| SDPA flash | PyTorch | Yes | **C** | \(4 h S^2 d_h + 5 h S^2\) | recompute | \(\Theta(hS)\) stats | `region/gqa/sdpa-flash` |
| SDPA mem_efficient | PyTorch | Yes | **C** | same as flash | recompute | no dense \(P\) | `region/gqa/sdpa-mem-efficient` |
| FlashAttention-2 | flash-attn | Yes | **C** | same | recompute | same | `region/gqa/flash2` |
| HF eager | transformers | No | A+GEMM | same as math | save \(P\) | identity | unfused |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gqa
status: to_implement
recommended_region_ids:
  - impl_id: region/gqa/sdpa-math
    variant: sdpa-math
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - impl_id: region/gqa/sdpa-flash
    variant: sdpa-flash
    hardware_gate: any
    fusion_boundary: C
    status: to_implement
  - impl_id: region/gqa/sdpa-mem-efficient
    variant: sdpa-mem-efficient
    hardware_gate: any
    fusion_boundary: C
    status: to_implement
pattern_rule:
  op_families:
    - repeat_kv
    - repeat_kv
    - transpose
    - matmul
    - add
    - multiply
    - exp
    - reduce_sum
    - divide
    - matmul
selection:
  attention_backend: sdpa
  sdpa_mode_state_key: sdpa_mode
  sdpa_mode_values: [math, flash, mem_efficient]
recipe:
  sdpa_math:
    forward_flops: "4 * h * S * S * d_h + 6 * h * S * S"
    backward_flops: "8 * h * S * S * d_h + 10 * h * S * S"
    save_P: true
    save_row_stats: false
    elided_temps: []
    saved_backward: [P]
    resource_events_forward: [ALLOCATE(scores), ALLOCATE(P), SAVE(P), ALLOCATE(context)]
  sdpa_flash:
    forward_flops: "4 * h * S * S * d_h + 5 * h * S * S"
    backward_flops: "8 * h * S * S * d_h + 10 * h * S * S"
    save_P: false
    save_row_stats: true
    elided_temps: [scores, masked_scores, scaled_scores, exp_scores, P, causal_mask]
    saved_backward: [row_stats]
    resource_events_forward: [ALLOCATE(context)]
  sdpa_mem_efficient:
    forward_flops: "4 * h * S * S * d_h + 5 * h * S * S"
    backward_flops: "8 * h * S * S * d_h + 10 * h * S * S"
    save_P: false
    save_row_stats: true
    elided_temps: [scores, P, causal_mask]
    saved_backward: [row_stats]
    resource_events_forward: [ALLOCATE(context)]
capabilities: [fused, sdpa]
priority: 7
```

Sibling leaves: `region/gqa/flash2`, `region/gqa/flash3` for `attention_backend=flash_attention_2`; decode `region/gqa/paged` (**G3**).

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Vaswani, A., et al. (2017). Attention is all you need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Rabe, M., & Staats, G. (2021). Self-attention does not need \(O(n^2)\) memory. [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)
- Dao, T., et al. (2022). FlashAttention. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- PyTorch SDPA: [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- PyTorch CUDA SDPA notes: [FlashAttention v2](https://pytorch.org/docs/stable/notes/cuda.html#flash-attention-v2)
- HF Transformers: [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- Metal-Flash SDPA: [`kernels-community/metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa)
- Local: [`docs/kernel-implementation.md`](docs/kernel-implementation.md) §11.2; [ADR-0009](docs/adr/0009-backend-profiles-and-attention-selection.md)

**Verification date:** 2026-09-08

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 7 | `region/gqa/sdpa-math` | §11.2 math row | A+GEMM | Explicit SDPA math sub-backend; materialized \(P\) |
| 7 | `region/gqa/sdpa-flash` | §11.2 flash row | C | SDPA flash dispatch; same VRAM elision as FA2 |
| 6 | `region/gqa/sdpa-mem-efficient` | §11.2 mem_efficient | C | xFormers-style tiled path |
| — | `attention_backend` field | ADR-0009 | — | Required selection axis before SDPA variants activate |
| 5 | `region/gqa/paged` | §11.9 | C (decode) | **G3** — separate work tile |
