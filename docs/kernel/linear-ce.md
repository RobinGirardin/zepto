# Zepto kernel research: FusedLinearCrossEntropy (Liger Kernel)

**Date:** 2026-09-08
**Proposer:** Robin Girardin
**Scope:** LM-head training cross-entropy fusion (boundary **D**); token-chunked GEMM + vocab softmax + CE; prefill primary; training; CUDA/ROCm (Liger Triton)
**Context:** `use_liger_kernel=True` replaces eager `Linear + CrossEntropyLoss` on the untied LM head; Zepto models `region/linear_ce/liger` as the default CUDA/ROCm leaf

---

## Section 0: Mathematical definition

Training cross-entropy on untied LM head hidden states \(H \in \mathbb{R}^{S \times d}\), weight \(W \in \mathbb{R}^{d \times V}\), integer labels \(y \in \{0,\ldots,V-1\}^S\):

\[
Z = H W \in \mathbb{R}^{S \times V}, \qquad
p_{sv} = \frac{e^{Z_{sv}}}{\sum_{v'} e^{Z_{sv'}}}, \qquad
\mathcal{L} = -\frac{1}{S}\sum_{s=1}^{S} \log p_{s,y_s}.
\]

**I/O shapes:** hidden \((S, d)\); weight \((d, V)\); labels \((S,)\); scalar loss output.

**Numerics policies:**
- Liger uses **online softmax** per token row inside each chunk (max-subtract on-chip; not billed as separate full-rank `ALLOCATE`s).
- Zepto softmax FLOP leaf for vocab axis bills **3 FLOPs/element** (exp + sum + div) per `docs/kernel-implementation.md` §14 — stable max/sub run on-chip in fused kernels.
- Labels are int64; loss scalar is fp32 reduction.

**Structural vs eager:** Eager PyTorch materializes full logits \((S,V)\) then `CrossEntropyLoss`. Liger streams token rows in chunks, computes partial GEMM tiles, online softmax, and CE without resident full \((S,V)\) logits.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | Liger `LigerFusedLinearCrossEntropyLoss` ([`fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)); TRL Hub [`trl-lib/fused-linear-ce`](https://huggingface.co/trl-lib/fused-linear-ce) (vocab-tile variant) |
| **Identity lowering** | Unfused semantic chain | `LinearMatMul` → `Exp` → `ReduceSum` → `Divide` → `Gather` → `Log` → `Multiply` → `ReduceSum` (Zepto `FusedLinearCrossEntropy` module) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/linear_ce/liger` (default CUDA/ROCm), `region/linear_ce/reference`, `region/linear_ce/hub-trl` |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **PyTorch eager** | `nn.Linear` + `CrossEntropyLoss` | No | D (unfused) | Any | Training |
| **Liger Kernel** | `LigerFusedLinearCrossEntropyLoss` via `use_liger_kernel=True` | Yes | **D** | CUDA, ROCm Triton | Training |
| **TRL Hub** | `get_kernel("trl-lib/fused-linear-ce")` | Yes | **D** | CUDA wheels | Training |
| **Zepto (registered)** | `region/linear_ce/*` | Yes (cost leaf) | **D** | hardware-gated variants | Training |

Liger tiles **token rows** (`chunk_size × V` logits live); TRL Hub tiles **vocabulary** (fp32 LSE accumulator per token). Both elide full \((S,V)\) resident logits.

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **D** — LM-head GEMM + vocab-axis softmax + cross-entropy; orthogonal to attention boundaries A/B/C.

**Mutually exclusive:**
- `region/linear_ce/liger` vs `region/linear_ce/hub-trl` on the same LM-head invocation — device-routed variants, same math leaf with different peak-tile formulas.
- `region/linear_ce` vs unfused `region/linear` + standalone softmax — fused region wins when `requested_capabilities={'fused'}` and pattern matches.
- Boundary **D** vs `region/masked_softmax` (B) — different softmax axis (vocab \(V\) vs keys \(S\)).

**Composable:**
- Liger RMSNorm / SwiGLU / CE patches compose per layer with FlashAttention Hub kernels ([TRL kernels hub](https://huggingface.co/docs/trl/en/kernels_hub)).
- Inference LM-head still uses `region/linear` (explicit logits for sampling); fused CE is **training-only**.

**Execution constraints:**
- Liger: `CHUNK_MEM_CONST = 16`; CUDA/ROCm; not routed on XPU/MPS for Liger CE.
- TRL Hub: CUDA-only; vocab-tile peak \(S \times V_{\mathrm{tile}} \times 4\) fp32.
- `requires_grad=False` → backward FLOPs = 0; use `LMHead` + `region/linear` for inference.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(S\) = tokens, \(d\) = hidden, \(V\) = vocab, \(|Z| = S V\).

### 4.1 Identity lowering (unfused eager)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | `LinearMatMul` | \((S,d)\times(d,V)\) | \(2 S d V\) | \(2 S d V\) |
| 2 | `Exp` | \((S,V)\) | \(S V\) | \(S V\) |
| 3 | `ReduceSum` | \((S,V)\) | \(S V\) | \(S V\) |
| 4 | `Divide` | \((S,V)\) | \(S V\) | \(S V\) |
| 5 | `Gather` + `Log` + `Multiply` + `ReduceSum` | per token | \(\approx 2 S\) | \(\approx 2 S\) |
| **Identity total** | | | | **\(2 S d V + 3 S V + 2 S\)** |

Full \((S,V)\) logits **materialized** at peak.

### 4.2 Fused region leaf (Liger kernel-accurate)

| Step | Primitive | Effective tile | FLOPs | Subtotal |
|------|-----------|----------------|-------|----------|
| 1 | Chunked GEMM | streaming | \(2 S d V\) | \(2 S d V\) |
| 2 | Online vocab softmax | per row | \(3 S V\) | \(3 S V\) |
| 3 | CE reduction | per token | \(2 S\) | \(2 S\) |
| **Fused total** | | | | **\(2 S d V + 3 S V + 2 S\)** |

Arithmetic unchanged vs eager; fusion elides **memory**, not FLOPs.

**Closed form (kernel-accurate):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 2 S d V + 3 S V + 2 S
\]

**Arithmetic intensity (numeric example):** Apertus-8B, \(S{=}8192\), \(d{=}4096\), \(V{=}131072\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| GEMM FLOPs | \(2 \times 8192 \times 4096 \times 131072 \approx 8.8 \times 10^{12}\) |
| Softmax FLOPs | \(3 \times 8192 \times 131072 \approx 3.2 \times 10^9\) |
| Min HBM (fused Liger peak logits) | \(\min(SVe,\; C S d e) = 1.0\) GiB with \(C{=}16\) |
| Vocab softmax AI | \(3/(2e) = 0.75\) FLOP/byte |

---

## Section 5: Backward FLOPs — step-by-step derivation

Liger **recomputes** each logits chunk in backward rather than saving full \((S,V)\).

| Step | Primitive | FLOPs | Subtotal |
|------|-----------|-------|----------|
| 1 | Recompute chunked GEMM + softmax forward | \(2 S d V + 3 S V\) | \(2 S d V + 3 S V\) |
| 2 | Softmax + CE VJP | \(\approx 4 S V\) | \(4 S V\) |
| 3 | GEMM backward (grad to \(H\), \(W\)) | \(\approx 4 S d V\) | \(4 S d V\) |
| **Backward total** | | | **\(\approx 6 S d V + 7 S V\)** |

`requires_grad=False` → backward FLOPs = **0**.

**Closed form (training, kernel-accurate recompute path):**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} \approx 6 S d V + 7 S V
\]

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Eager unfused** | full logits \((S,V)\) | \(S V e\) | — |
| **Liger \(C{=}16\)** | one chunk \((\mathrm{chunk\_size}, V)\) | \(\min(SVe,\; C S d e)\) | full \((S,V)\) logits |
| **Liger \(C{=}1\) (old floor)** | minimal chunk | \(\approx S d e\) | full logits |
| **TRL Hub vocab-tile** | tile accumulator | \(S \times V_{\mathrm{tile}} \times 4\) fp32 | full logits |

Liger chunk sizing ([source](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)):

```text
C = CHUNK_MEM_CONST = 16
inc_factor = cdiv(V, C * H)
chunk_size = next_power_of_2(cdiv(BT, inc_factor))
peak_logits = chunk_size * V * e
```

Apertus-8B (\(BT{=}8192\), \(H{=}4096\), \(V{=}131072\), \(C{=}16\)): \(\mathrm{chunk\_size}{=}4096\), peak logits **1.0 GiB** (not 2.0 GiB eager, not 0).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| **Eager unfused** | logits or CE intermediates | \((S,V)\) | \(S V e\) | full logits SAVE |
| **Liger fused** | hidden \(H\), weight \(W\) (autograd) | \((S,d)\), \((d,V)\) | \(S d e + V d e\) | SAVE hidden if grad; weight PERSIST |
| **Liger fused** | logits | — | **0** (recompute) | no logits SAVE |
| **Zepto fused leaf** | hidden only | \((S,d)\) | \(S d e\) | `save_hidden: true` when grad |

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
PERSIST(weight) → ALLOCATE(logits[S,V]) → ALLOCATE(probs[S,V]) → SAVE(logits) → ALLOCATE(loss_scalar)
```

**Fused region leaf (Liger / reference):**
```
PERSIST(weight) → ALLOCATE(logits_chunk[chunk_size,V]) → ALLOCATE(loss_scalar) → SAVE(hidden) [if grad]
```

**Apertus-8B trace (fused Liger, training, \(S{=}8192\)):**
- Peak logits chunk: **1.0 GiB**
- Saved hidden: \(8192 \times 4096 \times 2 =\) **64 MiB**
- Weight persistent: \(4096 \times 131072 \times 2 \approx\) **1.0 GiB**
- Elided: full 2.0 GiB \((S,V)\) logits

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| PyTorch eager | ATen | No | D | \(2SdV + 3SV\) | saves logits | full \((S,V)\) | identity |
| Liger FLCE | liger-kernel | Yes | D | \(2SdV + 3SV\) | recompute chunks | \(\min(SVe, CSde)\) | `region/linear_ce/liger` |
| TRL Hub | trl-lib/fused-linear-ce | Yes | D | \(2SdV + 3SV\) | vocab-tile recompute | fp32 tile accum | `region/linear_ce/hub-trl` |
| Zepto (registered) | — | Yes | D | same | recompute | chunk peak | `region/linear_ce/*` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: linear_ce
recommended_region_ids:
  - id: region/linear_ce/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: D
    status: registered
  - id: region/linear_ce/liger
    variant: liger
    hardware_gate: exclude_xpu_mps
    fusion_boundary: D
    status: registered
  - id: region/linear_ce/hub-trl
    variant: hub-trl
    hardware_gate: cuda_only
    fusion_boundary: D
    status: registered
pattern_rule:
  op_families:
    - linear_matmul
    - exp
    - reduce_sum
    - divide
    - gather
    - log
    - multiply
    - reduce_sum
recipe:
  forward_flops: "2*S*d*V + 3*S*V + 2*S"
  backward_flops: "6*S*d*V + 7*S*V (recompute path)"
  chunk_mem_const: 16
  materialize_full_logits: false
  save_logits: false
  save_hidden: true
  elided_temps: [logits_full, probs_full]
  saved_backward: [hidden]
  resource_events_forward:
    - PERSIST(weight)
    - ALLOCATE(logits_chunk)
    - ALLOCATE(loss_scalar)
    - SAVE(hidden)
  numerics_tags: [stable_fp32_online, vocab_axis]
capabilities: [fused]
priority: 8
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Dai et al., Liger-Kernel: [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
- Liger FLCE impl: [fused_linear_cross_entropy.py](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- Liger transformer wrapper: [fused_linear_cross_entropy.py (transformers)](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py)
- TRL Hub fused-linear-ce: [huggingface.co/trl-lib/fused-linear-ce](https://huggingface.co/trl-lib/fused-linear-ce)
- Wijmans et al., Cut Your Losses: [arXiv:2404.14280](https://arxiv.org/abs/2404.14280)
- TRL kernels hub docs: [huggingface.co/docs/trl/en/kernels_hub](https://huggingface.co/docs/trl/en/kernels_hub)
- Zepto authority: `docs/kernel-implementation.md` §14, §14.1

**Verification date:** 2026-09-08

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 4 | `region/linear_ce` | §14 + §14.1 | D | LM-head training CE; elides 2 GiB logits at Apertus-8B prefill |

**Open gaps:**
- **G4** implementation order: linear_ce follows GQA backend variants.
- TRL Hub vocab-tile peak formula needs runtime \(V_{\mathrm{tile}}\) from wheel — use conservative default in Zepto until probed.
- Inference LM-head (`LMHead` module) stays on `region/linear`; no fused CE at decode.
