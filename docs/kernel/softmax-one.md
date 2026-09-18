# Zepto kernel research: Megatron SoftmaxOne (attention-sink softmax)

**Date:** 2026-09-16
**Proposer:** Robin Girardin
**Scope:** Fusion boundary **A** (standalone row softmax with per-head sink denominator) and **B** composition (scale + mask + SoftmaxOne inside Megatron `FusedScaleMaskSoftmax` PyTorch fallback); prefill primary; training and inference; CUDA primary (Megatron stack); kernel-accurate Zepto leaf
**Context:** Attention score softmax with learnable or fixed denominator offset (“softmax off by one”); GPT-OSS / Megatron training path; not full FlashAttention boundary C (see cross-ref `region/gqa-sink`)

---

## Section 0: Mathematical definition

Per query row \(i\), head \(h\), key logits \(z_{h,i,j}\) for \(j \in \{1,\ldots,S\}\), and per-head sink logit \(s_h\) (learnable parameter or fixed scalar):

**Learnable sink (Megatron `SoftmaxOne`, GPT-OSS `sinks`):**

\[
p_{h,i,j} = \frac{\exp(z_{h,i,j})}{\exp(s_h) + \sum_{k=1}^{S} \exp(z_{h,i,k})}
= \frac{\exp(z_{h,i,j})}{\sum_{k=1}^{S+1} \exp(\tilde{z}_{h,i,k})}
\]

where the extended row is \(\tilde{z}_{h,i} = [z_{h,i,1},\ldots,z_{h,i,S}, s_h]\) and output drops the sink column.

**Fixed offset +1 (Evan Miller `softmax_1`, default `denominator_offset=1.0`):**

\[
p_{h,i,j} = \frac{\exp(z_{h,i,j})}{1 + \sum_{k=1}^{S} \exp(z_{h,i,k})}
\]

Equivalent to appending a synthetic key with logit \(\log 1 = 0\).

**I/O shapes:**
- Input scores \(Z\): \((B, h, S_q, S_k)\) or Zepto rank-3 \((h, S, S)\) after \(QK^\top\) and optional scale/mask
- Sink parameter: \((h,)\) per query head (`denominator_offset` / `sinks` / `attention_sink`)
- Output weights \(P\): same shape as \(Z\) (sink column discarded)

**Megatron reference implementation** ([`SoftmaxOne`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py)): broadcast sink to \((B,h,S_q,1)\), `torch.cat` along keys → \((B,h,S_q,S_k{+}1)\), `torch.softmax(dim=-1)`, slice `[..., :-1]`. **Not a dedicated CUDA kernel** — PyTorch ATen softmax on the extended row.

**GPT-OSS eager** uses the same concat pattern ([`eager_attention_forward`](https://github.com/huggingface/transformers/blob/c472755e/src/transformers/models/gpt_oss/modeling_gpt_oss.py)) with optional pre-softmax row max-subtract for bf16 stability.

**Numerics policies (bytes, not FLOPs):**
- Megatron `FusedScaleMaskSoftmax` may upcast to fp32 before scale/mask when `softmax_in_fp32=True` — doubles softmax tile bytes transiently; **0 extra FLOPs**.
- Sink is typically bf16/fp16 alongside scores; concat extends the softmax row by **one column** (\(+h S e\) bytes in the extended tile if materialized).
- Structural causal mask in Flash/Flex paths is **boundary C** — different region (`region/gqa-sink`); sink renormalization via LSE in Flex Attention ([`flex_attention.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py)).

**Zepto identity reference:** `AttentionSoftmaxWithSink` semantic op + module ([`attention_softmax_with_sink.py`](../../src/zepto/modules/attention_softmax_with_sink.py), [`semantic/operations/attention_softmax_with_sink.py`](../../src/zepto/semantic/operations/attention_softmax_with_sink.py)) — one fused op replacing decomposed `exp → reduce_sum → divide`.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production baseline for sink softmax | Megatron `SoftmaxOne` → `torch.cat` + `aten::softmax` on \((\cdot,\cdot,S{+}1)\) tile; GPT-OSS eager concat path; Megatron `FusedScaleMaskSoftmax.forward_torch_softmax(..., softmax_offset=...)` when fused CUDA kernels unavailable |
| **Identity lowering** | Unfused semantic chain | `Concat(sink)` → `Softmax` (or `Exp` → `ReduceSum` → `Add(sink)` → `Divide`); each intermediate full-rank \((h,S,S)\) or \((h,S,S{+}1)\) `ALLOCATE` |
| **Zepto fused region** | Kernel-accurate cost leaf | `AttentionSoftmaxWithSink` semantic op today; planned `region/softmax-one/*` (boundary A) and `region/masked_softmax` + sink variant (boundary B); boundary C cross-ref `region/gqa-sink/*` |

**Default estimates** use the Megatron eager concat+softmax reference or the Zepto fused leaf — not a sum of unfused primitives without labeling.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Megatron `SoftmaxOne`** | [`megatron.core.fusions.fused_softmax.SoftmaxOne`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py) | No (PyTorch cat + ATen softmax) | **A** | CUDA (via Megatron training stack); any device PyTorch supports | Both; autograd via `torch.softmax` |
| **Megatron `FusedScaleMaskSoftmax` + sink** | Same file; `forward_torch_softmax(..., softmax_offset)` | Partial (scale/mask separate; sink via SoftmaxOne) | **B** | CUDA; fused CUDA path **disabled** when `softmax_offset is not None` | Training primary |
| **GPT-OSS eager** | [`modeling_gpt_oss.eager_attention_forward`](https://github.com/huggingface/transformers/blob/c472755e/src/transformers/models/gpt_oss/modeling_gpt_oss.py) | No (matmul + mask + cat + softmax) | **C** unfused / **A** softmax sub-leaf | Any | Both |
| **GPT-OSS Flex Attention + LSE renorm** | [`integrations/flex_attention.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py) | Yes (attention + post LSE sink renorm) | **C** | CUDA (Flex); CPU Flex **unsupported** with sinks | Both |
| **PyTorch `F.softmax`** | ATen on extended \((S{+}1)\) row | Yes (softmax only) | A | CUDA/ROCm/MPS | Both |
| **Megatron CUDA masked softmax** | `scaled_masked_softmax_cuda` | Yes | B | CUDA | Training — **no sink**; mutually exclusive with SoftmaxOne path |
| **FlashAttention / `region/gqa-sink`** | Online softmax + sink in fused attention | Yes | **C** | CUDA/ROCm/MPS | Both |
| **Zepto (today)** | `AttentionSoftmaxWithSink` semantic op | Yes (cost leaf) | A | any | Forward only (`backward supported=False`) |
| **Zepto (registered C)** | `region/gqa-sink/flash2`, `flash3` | Yes | C | cuda / any | Both (recompute backward) |
| **Zepto (planned A/B)** | `region/softmax-one/*`, `region/masked_softmax/sink` | Yes (leaf) | A / B | any / cuda | TBD |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** for the SoftmaxOne leaf on a materialized \((h,S,S)\) scores tile; **B** when composed inside Megatron `FusedScaleMaskSoftmax` (scale + additive mask + SoftmaxOne fallback).

**Mutually exclusive:**
- `region/softmax-one/*` (A) vs `region/gqa-sink/*` (C) on the **same attention layer** — C replaces \(QK^\top\) + sink softmax + \(PV\).
- Megatron fused CUDA masked softmax (`scaled_*_softmax_cuda`) vs SoftmaxOne path — `is_kernel_available` returns false when `softmax_offset is not None`; only one path runs.
- `region/softmax-one/*` vs plain `region/softmax` (A without sink) on the same logits tile.

**Composable:**
- Boundary **B:** upstream `MatMul(Q,K^T)` + optional mask add → SoftmaxOne → downstream `MatMul(P,V)`.
- GPT-OSS Flex: boundary C attention kernel + post-hoc LSE sink renorm (extra \(\Theta(h S)\) elementwise work, not a separate \((h,S,S)\) tile).
- Liger patches (RMSNorm, CE) compose with `region/gqa-sink`; they do not provide a standalone SoftmaxOne leaf.

**Execution constraints:**
- Megatron SoftmaxOne requires materialized score logits; incompatible with SDPA `math` path that hides logits unless decomposed.
- GPT-OSS docs: SDPA unsupported with sinks; use Flash or Flex Attention.
- Sink parameter shape \((h,)\) must broadcast to query heads; GQA repeat-KV happens **before** softmax in GPT-OSS eager.
- Causal / sliding-window masks apply to score columns only; sink column is **unmasked** (always participates in denominator).

---

## Section 4: Forward FLOPs — step-by-step derivation

Tile: \(h\) heads, \(S\) query rows, \(S\) key columns. \(|tile| = h S^2\). Extended softmax tile \(|tile_+| = h S (S{+}1)\).

### 4.1 Identity lowering (Megatron eager concat + decomposed softmax)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Broadcast sink | reshape/expand | \((h,S,1)\) | 0 (indexing) | 0 |
| 2. Concat | append sink column | \((h,S,S{+}1)\) | 0 (layout; HBM copy billed in §6) | 0 |
| 3. Row max | stable max | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 4. Subtract | \(z - m\) | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 5. Exp | | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 6. Sum | row sum | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 7. Divide | normalize | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 8. Slice | drop sink column | \((h,S,S)\) | 0 | 0 |
| **Unfused total** | | | | **\(5 h S (S{+}1) = 5 h S^2 + 5 h S\)** |

Alternative closed-form denominator-only identity (no concat materialization, Zepto-style decomposition): `exp` + `sum` + `add(sink)` + `div` → **\(3 h S^2 + h S\)** — omits stable max-subtract; not numerically identical to ATen.

### 4.2 Fused region leaf (kernel-accurate)

**Reference path (Megatron/GPT-OSS eager):** ATen stable softmax on extended row — bill **\(5 h S (S{+}1)\)**.

**Zepto semantic leaf (`AttentionSoftmaxWithSink`):** models sink as denominator increment without materializing \((S{+}1)\) column:

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1–5. Stable softmax | max+sub+exp+sum+div | \((h,S,S)\) | \(5 h S^2\) | \(5 h S^2\) |
| 6. Sink denominator | add \(\exp(s_h)\) to row sum | \(h S\) rows | \(h S\) | \(h S\) |
| **Zepto fused total** | | | | **\(5 h S^2 + h S\)** |

On-chip max/sub in ATen are included in the **\(5 \cdot |tile|\)** billing for the reference path; they do not add separate `ALLOCATE` events.

**Boundary B add-on** (Megatron torch fallback before SoftmaxOne): scale \(+\) mask add → **\(+2 h S^2\)** FLOPs (same as `region/masked_softmax` without sink).

### 4.3 Paper-comparable (if different)

Evan Miller [`softmax_1`](https://www.evanmiller.org/attention-is-off-by-one.html): denominator \(1 + \sum \exp(z)\) — **paper-comparable** forward **\(3 h S^2\)** (exp + sum + div only, no max-subtract, fixed \(+1\) not learnable \(\exp(s)\)).

Appendix E standard attention softmax: **\(3 h S^2\)** — label **paper-comparable only**.

**Closed form (kernel-accurate, Megatron reference):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{ref}} = 5\, h S (S + 1) = 5 h S^2 + 5 h S
\]

**Closed form (Zepto fused leaf):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{zepto}} = 5 h S^2 + h S
\]

**Arithmetic intensity (numeric example):** Apertus-8B prefill, \(h{=}32\), \(S{=}8192\), bf16 (\(e{=}2\)):

| Quantity | Megatron ref | Zepto leaf |
|----------|--------------|------------|
| FLOPs | \(5 \times 32 \times 8192 \times 8193 \approx 1.072 \times 10^{10}\) | \(5 \times 32 \times 8192^2 + 32 \times 8192 \approx 1.071 \times 10^{10}\) |
| HBM (ideal 1-pass: read \(Z\), write \(P\)) | \(\approx 2 h S^2 e = 4.0\) GiB | same |
| AI (1-pass bound) | \(\approx 1.25\) FLOP/byte | \(\approx 1.25\) FLOP/byte |

Extended concat path may read/write \((h,S,S{+}1)\) once → **\(\approx 4.0005\) GiB** transient peak for the extended tile (+0.012% vs \(S^2\) tile).

---

## Section 5: Backward FLOPs — step-by-step derivation

Megatron `SoftmaxOne` delegates backward to `torch.softmax` on the extended tensor (autograd through `cat` + slice).

VJP for softmax row (saved output \(P'\) including sink column before slice, or recomputed):

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1. Row dot | \(\langle P', dY'\rangle\) | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 2. Subtract | | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 3. Multiply | \(P' \odot \ldots\) | \((h,S,S{+}1)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| 4. Chain / slice adjoint | map to \(dZ\) | \((h,S,S)\) | \(h S (S{+}1)\) | \(h S (S{+}1)\) |
| **Total (ref)** | | | | **\(\approx 4 h S (S{+}1) = 4 h S^2 + 4 h S\)** |

Sink parameter gradient: \(\sum_{i} \partial \mathcal{L} / \partial s_h\) — \(\Theta(h S)\) reductions; folded into extended-softmax backward.

**Zepto today:** `AttentionSoftmaxWithSink.backward` → `BackwardSpec(supported=False)` → **backward FLOPs = 0** in estimator (gap).

**Closed form (reference training path):**
\[
\mathrm{FLOPs}_{\mathrm{bwd}}^{\mathrm{ref}} \approx 4 h S (S + 1)
\]

(`requires_grad=False` → backward FLOPs = 0.)

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

Worked unit: one \((h,S,S)\) bf16 buffer, \(h{=}32\), \(S{=}8192\) = **4.0 GiB**.

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Megatron SoftmaxOne (eager)** | `scores`, `combined_logits` \((h,S,S{+}1)\), `probs` (pre-slice), `P` | **8–12 GiB** (2–3× tile; extended +1 column) | — |
| **GPT-OSS eager full attention** | QK scores + extended logits + weights + context | **G4b false peak** if summed naively | — |
| **Zepto identity (decomposed)** | `scores`, `exp_scores`, `denom`, `P` | **8–12 GiB** | — |
| **Zepto fused leaf (target `region/softmax-one`)** | **`P` only** | **4.0 GiB** | extended logits, `exp_scores`, on-chip max/sub |
| **Flash `region/gqa-sink` (boundary C)** | context \((h,S,d_h)\); optional `row_stats` | **no \((h,S,S)\)** | entire score tile |

Persistent sink weights: \((h,)\) — **\(2h\) bytes** bf16 (negligible vs activations).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Megatron SoftmaxOne / ATen | Softmax output **before slice** (incl. sink col) or equivalent | \((h,S,S{+}1)\) | \(\Theta(h S^2)\) | `save_P_extended: true` |
| Zepto semantic (today) | none | — | 0 | backward unsupported |
| `region/gqa-sink` Flash | Row stats \((m,\ell)\) | \((h,S,2)\) | \(\Theta(h S)\) | `save_row_stats: true` when grad |

### 6.3 Resource event chains

**Identity lowering (Megatron concat path):**
```
ALLOCATE(scores) → ALLOCATE(combined_logits) → ALLOCATE(P_full) → ALLOCATE(P) → SAVE(P_full or P)
```

**Zepto fused region leaf (`region/softmax-one/reference`, target):**
```
ALLOCATE(P) → SAVE(P)   # training; sink param is PERSIST
```

**Boundary B (Megatron torch fallback: scale + mask + SoftmaxOne):**
```
ALLOCATE(P) → SAVE(P)   # elides scaled_scores, masked_scores, combined_logits when fused leaf modeled
```

**Boundary C cross-ref (`region/gqa-sink/flash2`):**
```
ALLOCATE(output) → [SAVE(row_stats) if grad]
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Megatron SoftmaxOne | megatron.core | No (ATen) | A | \(5hS(S{+}1)\) | \(4hS(S{+}1)\) | Extended tile + \(P\) | `region/softmax-one/megatron` |
| Megatron B + sink | FusedScaleMaskSoftmax fallback | Partial | B | \(7hS^2 + 5hS\) | \(\approx 4hS(S{+}1)\) | 2–3× tile peak | `region/masked_softmax/sink` |
| GPT-OSS eager | transformers | No | A/C | \(5hS(S{+}1)\) sub-leaf | \(4hS(S{+}1)\) | Extended concat | identity |
| GPT-OSS Flex + LSE | transformers | Yes | C | §11 gqa-sink | recompute | LSE + renorm | `region/gqa-sink/*` |
| Zepto semantic | zepto | Cost leaf | A | \(5hS^2 + hS\) | 0 (unsupported) | `ALLOCATE(P)` | op only |
| Standard softmax | ATen | Yes | A | \(5hS^2\) | \(4hS^2\) | \(P\) only | `region/softmax` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: softmax_one
recommended_region_ids:
  - id: region/softmax-one/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/softmax-one/megatron
    variant: megatron
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/masked_softmax/sink
    variant: megatron_torch_fallback
    hardware_gate: cuda_only
    fusion_boundary: B
    status: to_implement
  - id: region/gqa-sink/flash2
    variant: flash2
    hardware_gate: any
    fusion_boundary: C
    status: registered
  - id: region/gqa-sink/flash3
    variant: flash3
    hardware_gate: cuda_only
    fusion_boundary: C
    status: registered
pattern_rule:
  op_families:
    - attention_softmax_with_sink
recipe:
  forward_flops: "5 * h * S * S + h * S  # Zepto fused leaf; ref Megatron eager: 5 * h * S * (S + 1)"
  backward_flops: "4 * h * S * (S + 1) if requires_grad else 0  # ref; Zepto semantic backward unsupported today"
  forward_flops_per_element: 5
  backward_flops_per_element: 4
  materialize_extended_logits: false
  save_P: true
  save_P_extended: false
  elided_temps:
    - combined_logits
    - exp_scores
    - row_max
    - shifted_logits
    - sink_column
  saved_backward:
    - name: P
      shape: "(h, S, S) or (B, h, S, S)"
  resource_events_forward:
    - "ALLOCATE(P) → SAVE(P)"
  resource_events_backward:
    - "RELEASE(P) on backward phase end"
  numerics_tags:
    - stable_max_subtract
    - attention_sink
    - softmax_off_by_one
    - megatron_softmax_one
capabilities:
  - fused
  - attention_sink
priority: 5
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Evan Miller, “Attention Is Off By One” (softmax\_1 definition): https://www.evanmiller.org/attention-is-off-by-one.html
- Megatron-LM `SoftmaxOne` and `FusedScaleMaskSoftmax`: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py
- GPT-OSS eager attention with sinks: https://github.com/huggingface/transformers/blob/c472755e/src/transformers/models/gpt_oss/modeling_gpt_oss.py
- GPT-OSS Flex Attention sink renormalization: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
- GPT-OSS model docs (sink requirement, SDPA unsupported): https://huggingface.co/docs/transformers/en/model_doc/gpt_oss
- Zepto `AttentionSoftmaxWithSink` semantic op: `src/zepto/semantic/operations/attention_softmax_with_sink.py`
- Zepto GQA-sink archived research: `docs/kernel/gqa-sink-softmax.md`
- Zepto domain doc §10 (Megatron catalog row): `docs/kernel-implementation.md`

**Verification date:** 2026-09-16

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P1 | `region/softmax-one/reference` | §8 | A | Standalone sink-softmax leaf; replaces decomposed exp/sum/div; fixes G4b on GPT-OSS/Megatron score path |
| P2 | `region/softmax-one/megatron` | §8 | A | Match Megatron eager FLOP leaf \(5hS(S{+}1)\) when profiling Megatron training stack |
| P3 | `region/masked_softmax/sink` | §8 | B | Scale+mask+SoftmaxOne torch fallback inside `FusedScaleMaskSoftmax` |
| — | semantic backward | §5 | A | `AttentionSoftmaxWithSink.backward supported=False` — training bwd gap |
| — | G4b | §6.1 | — | Unfused identity + concat inflates peak VRAM on score sub-chain |

**Open gaps:**
- **Backward:** Zepto semantic op bills `backward_flops=0`; reference Megatron path uses extended-softmax autograd (~\(4hS(S{+}1)\)).
- **FLOP reconciliation:** Zepto fused leaf (\(5hS^2 + hS\)) vs Megatron concat reference (\(5hS^2 + 5hS\)) — architect should pick profile routing or unify formula with documented delta.
- **No CUDA fused SoftmaxOne:** Unlike `scaled_masked_softmax_cuda`, sink path always materializes extended row in reference implementations — fusion opportunity for future kernel.
- **Discovery wiring:** Pattern must match `AttentionSoftmaxWithSink` module / `attention_softmax_with_sink` family inside GQA graphs; mutual exclusion with `region/softmax` and `region/masked_softmax`.
- **Boundary C:** Full attention with sinks registered as `region/gqa-sink/*`; do not double-count softmax FLOPs when C region is selected.
