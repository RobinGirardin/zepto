# Zepto kernel research: FlashAttention-3

**Date:** 2026-09-08
**Proposer:** Robin Girardin
**Scope:** Fusion **boundary C** — full GQA prefill attention (\(QK^\top\) + online softmax + \(PV\)) on NVIDIA Hopper via FlashAttention-3. Training + inference; CUDA Hopper primary (H100/H800); kernel-accurate Zepto leaf (not paper-comparable Appendix E softmax alone).
**Context:** Apertus-8B defaults: \(B{=}1\), \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\), \(S{=}8192\), bf16 (\(e{=}2\)). FA3 shares FLOP/VRAM class with FA2; distinct packaging and Hopper hardware gate.

**Known Zepto gaps flagged:** **G4b** (unfused GQA false peak if Flash region not selected); **G3** (decode/paged KV is a sibling leaf, not this report).

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

Per query position \(i\) and key \(j\):

\[
z_{ij}=\frac{(q_i k_{\pi(i),j}^\top)}{\sqrt{d_h}},\qquad
m_i=\max_j z_{ij},\qquad
p_{ij}=\frac{e^{z_{ij}-m_i}}{\sum_k e^{z_{ik}-m_i}},\qquad
o_i=\sum_j p_{ij}\, v_{\pi(i),j}.
\]

\(\pi(i)=\lfloor i\cdot h_{\mathrm{kv}}/h\rfloor\) maps query heads to KV heads. FlashAttention-2/3 **index** \(\pi(i)\) inside the kernel — no `repeat_kv` materialization ([Dao, 2023, §3.1.2](https://arxiv.org/abs/2307.08691)).

**FlashAttention-3** (Shah et al., [2024](https://arxiv.org/abs/2407.08608)) reimplements the **same tiling algorithm** as FA1/FA2 for Hopper: warp-specialized TMA + WGMMA pipelining, overlap of block GEMMs with online softmax, optional FP8 forward. Package: [`Dao-AILab/flash-attention` `hopper/`](https://github.com/Dao-AILab/flash-attention) (`flash_attn_3.flash_attn_interface`); Hub [`kernels-community/flash-attn3`](https://huggingface.co/kernels-community/flash-attn3).

**Numerics (bytes, not FLOPs):**
- Scale \(1/\sqrt{d_h}\) folded into GEMM or applied as scalar — **0 extra FLOPs** vs explicit multiply.
- Online softmax keeps row stats \((m_i,\ell_i)\) in **SRAM** during forward; dense \(P\) never written to HBM ([Dao et al., 2022, Thm. 1](https://arxiv.org/abs/2205.14135)).
- FP8 forward (FA3 beta): block-quantization scales add \(\Theta(\text{blocks})\) metadata; FP8 backward not in current beta — training backward uses FP16/BF16 path.
- Structural causal mask (`is_causal=True`): **0 mask bytes**, **0 mask-add FLOPs** in tile loop.

**Zepto identity reference:** `src/zepto/modules/gqa.py` — `RepeatKV → MatMul(QK^T) → Add(mask) → Softmax(scale→exp→sum→div) → MatMul(PV)`.

**Structural vs eager:** Boundary C replaces the entire score + context chain; projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) remain outside the fused leaf.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | FA3 Hopper: `flash_attn_3.flash_attn_func` / Hub `flash-attn3`; HF `attn_implementation="flash_attention_2"` often routes to FA2/FA3 family on Hopper |
| **Identity lowering** | Structural / debug chain | Zepto `GroupedQueryAttention`: `repeat_kv` ×2 → `matmul` → `add` → decomposed softmax (4 ops) → `matmul`; materializes \((h,S,S)\) scores and \(P\) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gqa/flash3` — bills \(4 h S^2 d_h + 5 h S^2\); **no** \((h,S,S)\) HBM temps; saves \(\Theta(hS)\) row stats for training backward |

**Default estimates** use the **fused region leaf** when `requested_capabilities` includes `fused` and `flash`.

**Two costs — never mixed:**
1. **Theoretical FLOPs** — GEMM + stable online softmax (fusion does not drop GEMM terms).
2. **HBM traffic / peak VRAM** — FA3 lowers bytes moved vs eager; same **asymptotic VRAM class** as FA2.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| HF Transformers eager | [`eager_attention_forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | No | A + GEMMs | All | Saves \(P\); fp32 softmax tile |
| PyTorch SDPA `flash` | [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Yes if selected | **C** | CUDA/ROCm | Same class as Flash when pinned |
| FlashAttention-1 | [Dao et al., 2022](https://arxiv.org/abs/2205.14135) | Yes | **C** | CUDA | Exact; \(\mathcal{O}(S)\) extra memory |
| FlashAttention-2 | [Dao, 2023](https://arxiv.org/abs/2307.08691); [`flash_attn_func`](https://github.com/Dao-AILab/flash-attention) | Yes | **C** | CUDA Ampere+ | GQA without `repeat_kv` |
| **FlashAttention-3** | [Shah et al., 2024](https://arxiv.org/abs/2407.08608); Hub [`flash-attn3`](https://huggingface.co/kernels-community/flash-attn3) | Yes | **C** | **Hopper** (CUDA ≥ 12.3) | Same VRAM class as FA2; ~1.5–2× wall-clock vs FA2 on H100 |
| FlashAttention-4 | Hub [`flash-attn4`](https://huggingface.co/kernels-community/flash-attn4) | Yes | **C** | Hopper/Blackwell | Same bf16 VRAM class |
| Hub `vllm-flash-attn3` | [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) | Yes | **C** | Hopper serving | Varlen packaging; same elision |
| Liger Kernel | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | No attention C | — | — | Composes **around** Flash; no C kernel |
| Metal-Flash SDPA | Hub [`metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) | Yes | **C** | MPS | Same VRAM class; not FA3 |
| Megatron/TE masked softmax | boundary **B** only | Partial | B | CUDA | Mutually exclusive with C |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **C** (full attention: \(QK^\top\) + online softmax + \(PV\))

**Mutually exclusive:**
- `region/gqa/flash*` (FA2, FA3, FA4, Metal-Flash, Sage) **replaces** `region/masked_softmax` and standalone `region/softmax` on the same attention layer.
- `region/gqa/paged` is decode-only (\(S_q{=}1\)) — different work tile; not interchangeable with prefill Flash.

**Composable:**
- Liger patches (RMSNorm, CE, SwiGLU) compose with FlashAttention Hub kernels per [TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub).
- Q/K/V/O linear projections remain separate `region/linear` leaves.

**Execution constraints:**
- FA3 requires Hopper GPU + CUDA ≥ 12.3 for the reference build.
- Causal masking is structural — no dense \((S,S)\) mask tensor when `is_causal=True`.
- GQA: K/V at \(h_{\mathrm{kv}}\) without `repeat_kv` copy in Flash path.

---

## Section 4: Forward FLOPs — step-by-step derivation

Tile: one layer, prefill, \(B{=}1\), GQA with \(h\) query heads, sequence \(S\), head dim \(d_h\). Score tile elements: \(|Z| = h S^2\).

### 4.1 Identity lowering (unfused eager)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | RepeatKV (K) | \((h_{\mathrm{kv}},S,d_h)\to(h,S,d_h)\) | 0 (copy) | 0 |
| 2 | RepeatKV (V) | same | 0 | 0 |
| 3 | MatMul \(QK^\top\) | \((h,S,d_h)\times(h,d_h,S)\) | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |
| 4 | Add (mask) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 5 | Multiply (scale) | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 6 | Exp | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 7 | ReduceSum | per row | \(h S^2\) | \(h S^2\) |
| 8 | Divide | \((h,S,S)\) | \(h S^2\) | \(h S^2\) |
| 9 | MatMul \(PV\) | \((h,S,S)\times(h,S,d_h)\) | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |

Identity subtotal (kernel-accurate softmax = 5 ops on scores after mask): \(4 h S^2 d_h + 6 h S^2\) (includes mask add).

### 4.2 Fused region leaf (FA2/FA3 kernel-accurate)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | Block GEMM \(QK^\top\) (tiled) | SRAM blocks | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |
| 2 | Online softmax (max+sub+exp+sum+div) | per row in tile | \(5 h S^2\) | \(5 h S^2\) |
| 3 | Block GEMM \(PV\) (tiled) | SRAM blocks | \(2 h S^2 d_h\) | \(2 h S^2 d_h\) |

Causal structural mask: **0** mask-add FLOPs billed in fused leaf (skipped in tile loop).

### 4.3 Paper-comparable (Appendix E)

Appendix E counts softmax as \(3 h S^2\) (exp + sum + div without stable max-subtract):

\[
\mathrm{FLOPs}_{\mathrm{fwd,paper}} = 4 h S^2 d_h + 3 h S^2.
\]

**Closed form (Zepto kernel-accurate leaf):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 4 h S^2 d_h + 5 h S^2.
\]

**Apertus-8B numeric example** (\(h{=}32\), \(S{=}8192\), \(d_h{=}128\)):
- GEMMs: \(4 \times 32 \times 8192^2 \times 128 = 1.073741824 \times 10^{12}\) FLOPs
- Softmax leaf: \(5 \times 32 \times 8192^2 = 1.073741824 \times 10^{10}\) FLOPs
- Total ≈ **1.0845 TFLOPs** per layer prefill

**Arithmetic intensity (fused leaf, minimum HBM bytes):**
- Inputs/outputs: Q, K, V, O at \(\Theta(h S d_h)\) with K/V at \(h_{\mathrm{kv}}\): ≈ \(2(h + 2 h_{\mathrm{kv}}) S d_h e\) bytes
- At \(S{=}8192\), bf16: ≈ **384 MiB** moved minimum vs **8–12 GiB** eager peak from \((h,S,S)\) temps
- AI ≈ \(1.08 \times 10^{12} / (384 \times 2^{20}) \approx 2700\) FLOPs/byte (order-of-magnitude; FA3 improves constant via tiling)

---

## Section 5: Backward FLOPs — step-by-step derivation

FlashAttention backward **recomputes** attention tiles from saved row statistics \((m,\ell)\), not stored \(P\) ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)). Recomputation adds another pass through \(QK^\top\) and softmax inside tiles.

| Step | Primitive | FLOPs (order) |
|------|-----------|---------------|
| Recompute \(QK^\top\) tiles | Block GEMM | \(2 h S^2 d_h\) |
| Recompute softmax VJP | Online softmax backward | \(5 h S^2\) |
| \(PV\) backward | Block GEMMs | \(4 h S^2 d_h\) |
| \(Q,K,V\) gradients | Additional GEMMs | \(4 h S^2 d_h\) |

**Closed form (training, FA2/FA3 recompute path):**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} \approx 2 \cdot (4 h S^2 d_h + 5 h S^2) = 8 h S^2 d_h + 10 h S^2.
\]

`requires_grad=False` (inference): backward FLOPs = **0**.

FP8 FA3 forward-only beta: backward uses FP16/BF16 FA3 path when enabled — same recompute accounting.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by FA3 fusion |
|------|--------------|-------------------|----------------------|
| Eager identity | scores, masked scores, fp32 softmax tile, \(P\) | \(\Theta(h S^2)\) — **4–8 GiB** per \((h,S,S)\) bf16 at \(S{=}8192\) | — |
| FA2/FA3 fused | context output only | \(\Theta(h S d_h)\) | scores, \(P\), fp32 tile, `repeat_kv` copies, causal mask |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Eager HF | \(P\) after softmax | \((h,S,S)\) | \(\Theta(h S^2)\) | `save_P=true` |
| FA2/FA3 | Row stats \((m,\ell)\) | \((h,S)\) fp32 ×2 | \(B h S \cdot 8\) | `save_row_stats=true` |

At \(h{=}32\), \(S{=}8192\): row stats ≈ **2 MiB** vs **4 GiB** for saved \(P\).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(k_repeated) → ALLOCATE(v_repeated) → ALLOCATE(scores) →
ALLOCATE(masked_scores) → ALLOCATE(P) → SAVE(P) → ALLOCATE(context)
```

**Fused region leaf (FA3):**
```
ALLOCATE(context) → SAVE(row_stats)   # training only; no (h,S,S) temps
```

Elided on-chip (not billed): SRAM tile buffers for Q/K/V blocks, online \((m,\ell)\) during forward, block FP8 scales (metadata only if FP8 forward).

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF eager GQA | transformers | No | A+GEMM | \(4 h S^2 d_h + 6 h S^2\) | save \(P\) | \((h,S,S)\) temps | identity |
| SDPA flash | PyTorch | Yes | **C** | \(4 h S^2 d_h + 5 h S^2\) | recompute | \(\Theta(hS)\) stats | `region/gqa/flash2` |
| FlashAttention-2 | flash-attn | Yes | **C** | same | recompute | no \(P\) | `region/gqa/flash2` |
| **FlashAttention-3** | flash-attn3 / hopper | Yes | **C** | **same as FA2** | **same as FA2** | **same as FA2** | **`region/gqa/flash3`** |
| FlashAttention-4 | flash-attn4 | Yes | **C** | same | same | same | `region/gqa/flash4` |
| vLLM FA3 | vllm-flash-attn3 | Yes | **C** | same | same | same + LSE workspace | `region/gqa/flash3` |
| Liger | — | — | — | — | — | no C kernel | compose only |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gqa
status: to_implement
recommended_region_ids:
  - impl_id: region/gqa/flash3
    variant: flash-attn3
    hardware_gate: cuda_only
    fusion_boundary: C
    status: to_implement
  - impl_id: region/gqa/flash2
    variant: flash-attn2
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
recipe:
  forward_flops: "4 * h * S * S * d_h + 5 * h * S * S"
  backward_flops: "8 * h * S * S * d_h + 10 * h * S * S"
  paper_comparable_forward_flops: "4 * h * S * S * d_h + 3 * h * S * S"
  materialize_P: false
  materialize_causal_mask: false
  materialize_repeat_kv: false
  save_P: false
  save_row_stats: true
  elided_temps:
    - scores
    - masked_scores
    - scaled_scores
    - exp_scores
    - P
    - k_repeated
    - v_repeated
    - causal_mask
  saved_backward:
    - row_stats
  resource_events_forward:
    - ALLOCATE(context)
  resource_events_backward:
    - SAVE(row_stats)
    - RELEASE(row_stats)
  numerics_tags:
    - causal_structural
    - gqa_implicit
    - online_softmax
    - hopper_fa3
capabilities:
  - fused
  - flash
  - gqa
priority: 10
```

Sibling leaves (not this invocation): `region/gqa/paged` for decode (\(S_q{=}1\)); `region/masked_softmax` for boundary B score-only fusion.

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Vaswani, A., et al. (2017). Attention is all you need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Ainslie, J., et al. (2023). GQA. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Dao, T., et al. (2022). FlashAttention. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, T. (2023). FlashAttention-2. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah, J., et al. (2024). FlashAttention-3. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608). [PyTorch blog](https://pytorch.org/blog/flashattention-3)
- Implementation: [Dao-AILab/flash-attention `hopper/`](https://github.com/Dao-AILab/flash-attention)
- Hub: [`kernels-community/flash-attn3`](https://huggingface.co/kernels-community/flash-attn3), [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3)
- HF Transformers: [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- Local: [`docs/kernel-implementation.md`](../docs/kernel-implementation.md) §11.5–§11.7; [`src/zepto/modules/gqa.py`](../src/zepto/modules/gqa.py)

**Verification date:** 2026-09-08

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 10 | `region/gqa/flash3` | this report; `docs/kernel-implementation.md` §11.5 | C (prefill) | Hopper FA3 packaging; same VRAM elision as FA2; fixes **G4b** false peak when `flash` capability set |
| 8 | `region/gqa/flash2` | §11.4 | C | Non-Hopper Flash fallback variant |
| 5 | `region/gqa/paged` | §11.9 | C (decode) | Separate work tile — **G3** |
