# Zepto kernel research: linear-ce-softcap (CappedFusedLinearCrossEntropy)

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** LM-head training path with **logit soft-capping before vocab softmax + CE** (boundary **D**); Muse Glimmer / Gemma 4 parity; prefill-length \(S\) token batch; training with autograd; CUDA/ROCm primary (Liger path)
**Context:** Checkpoints apply \(c \cdot \tanh(z/c)\) (plus optional output scale \(s\)) on raw logits **before** cross-entropy. Zepto module `CappedFusedLinearCrossEntropy` records the decomposed graph; fused region `region/linear_ce_softcap` is **planned** (Step 7 follow-up). Sub-regions `region/logit_softcap` and `region/linear_ce` are registered separately today.

---

## Section 0: Mathematical definition

Hidden states \(H \in \mathbb{R}^{S \times d}\), untied head weight \(W \in \mathbb{R}^{d \times V}\), labels \(y \in \{0,\ldots,V-1\}^S\), soft-cap \(c > 0\), optional output scale \(s > 0\) (Muse `output_multiplier`):

\[
Z = H W \in \mathbb{R}^{S \times V}, \qquad
\tilde{Z}_{sv} = s \cdot c \cdot \tanh\!\left(\frac{Z_{sv}}{c}\right),
\]

\[
p_{sv} = \frac{e^{\tilde{Z}_{sv}}}{\sum_{v'} e^{\tilde{Z}_{sv'}}}, \qquad
\mathcal{L} = -\frac{1}{S}\sum_{s=1}^{S} \log p_{s,y_s}.
\]

**I/O shapes:** \(H\,(S,d)\); \(W\,(d,V)\); labels \((S,)\); scalar loss.

**Checkpoint numerics (eager reference):**
- **Muse Glimmer:** pre-scale logits by `output_multiplier`, then Gemma-style cap with `final_logit_softcapping = T` → effectively \(T \cdot \tanh(\mathrm{mult}\cdot z / T)\) ([`modeling_muse_glimmer.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py) ~L1167–1172).
- **Gemma 4:** tied head then `final_logit_softcapping` tanh cap when set ([`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py) ~L1864–1867).

**Structural vs eager:** Zepto decomposes cap as `Divide` → `Tanh` → `Multiply` → `Multiply` under `LogitSoftCap` provenance (`src/zepto/modules/logit_soft_cap.py`), then the same 7-op CE chain as `FusedLinearCrossEntropy`. A **single fused region** should bill one kernel-accurate leaf for GEMM + cap + online vocab softmax + CE (Liger-style streaming), not a sum of three independent region wins on overlapping tensors.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | Liger `fused_linear_cross_entropy` with `softcap=` applied on each logits chunk before online softmax ([`fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)); HF eager `Linear` + tanh cap + `CrossEntropyLoss` |
| **Identity lowering** | Unfused semantic chain | `CappedFusedLinearCrossEntropy`: `LinearMatMul` → 4 cap ops → `Exp` → `ReduceSum` → `Divide` → `Gather` → `Log` → `Multiply` → `ReduceSum` (`src/zepto/modules/capped_fused_linear_cross_entropy.py`) |
| **Zepto fused region** | Kernel-accurate cost leaf | **Planned** `region/linear_ce_softcap/*` (Muse/Gemma training); **registered partials:** `region/logit_softcap/reference` (cap only), `region/linear_ce/*` (uncapped CE only) |

Default estimates should use the **fused reference** (Liger softcap + FLCE) or its Zepto `region/linear_ce_softcap` leaf — not double-count `region/linear_ce` and `region/logit_softcap` on the same logits tile when the parent region matches.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF Transformers eager** | Muse / Gemma `forward` logits path | No | D (cap + CE separate) | Any | Train + infer (infer keeps full logits) |
| **PyTorch eager CE** | `nn.Linear` + tanh cap + `CrossEntropyLoss` | No | D | Any | Training |
| **Liger FLCE + softcap** | `LigerFusedLinearCrossEntropyLoss(..., softcap=c)` | Yes (chunked) | **D** | CUDA / ROCm Triton | Training |
| **Liger FLCE (no cap)** | `use_liger_kernel=True` default | Yes | D | CUDA / ROCm | Training |
| **TRL Hub fused-linear-ce** | `trl-lib/fused-linear-ce` | Yes | D | CUDA | Training (no documented softcap knob in Zepto scope) |
| **Zepto module** | `CappedFusedLinearCrossEntropy` | Graph only | D | Any (identity lowering) | Training graph |
| **Zepto regions** | `region/logit_softcap`, `region/linear_ce` | Partial | D | hardware-gated CE variants | Training |
| **Zepto (planned)** | `region/linear_ce_softcap` | Yes (cost leaf) | **D** | inherit Liger gates | Training |

No public fused kernel merges cap with TRL Hub vocab-tile CE in one documented API; cost Muse/Gemma training with **Liger softcap path** or Zepto composite leaf.

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **D** — LM-head GEMM + **vocab-axis** tanh soft-cap + vocab softmax + cross-entropy. Orthogonal to attention boundaries A/B/C.

**Mutually exclusive:**
- `region/linear_ce_softcap/*` vs `region/linear_ce/*` on the same `CappedFusedLinearCrossEntropy` invocation when `requested_capabilities={'fused'}` and pattern matches — capped parent replaces uncapped CE region.
- `region/linear_ce_softcap/*` vs stacking `region/linear_ce` + `region/logit_softcap` on identical logits — **do not double-bill**; prefer parent region when implemented.
- Boundary **D** vs `region/masked_softmax` (B) — vocab axis \(V\) vs key axis \(S\).

**Composable:**
- Cap may appear on **inference** via `LanguageModelOutput` + `LogitSoftCap` without CE (`region/logit_softcap` + `region/linear`) — different module boundary from training FLCE.
- Liger training patches (RMSNorm, SwiGLU, FLCE) still compose with FlashAttention per [TRL kernels hub](https://huggingface.co/docs/trl/en/kernels_hub).

**Execution constraints:**
- Soft-cap constants \(c, s\) are **not learned** (`requires_grad=False` on cap tensors in Zepto); backward flows through \(\tanh\) into \(Z\) and GEMM when hidden/weight require grad.
- Liger applies cap **inside the chunk loop** before softmax (`HAS_SOFTCAPPING`); peak logits tile sizing unchanged vs uncapped FLCE (`CHUNK_MEM_CONST = 16`).
- Inference with sampling: use `LanguageModelOutput` (matmul + optional cap), not fused CE.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(|Z| = S V\). Soft-cap fused leaf uses **7 FLOPs/element** forward (`LogitSoftCapRecipe`, `src/zepto/analysis/lowering/recipes/logit_soft_cap.py`). Vocab softmax + CE uses **3 FLOPs/element + 2 S** (`LinearCERecipe`, `docs/kernel-implementation.md` §14).

### 4.1 Identity lowering (unfused, no parent region)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | `LinearMatMul` | \((S,d)\times(d,V)\) | \(2 S d V\) | \(2 S d V\) |
| 2 | `Divide` | \((S,V)\) | \(S V\) | \(S V\) |
| 3 | `Tanh` | \((S,V)\) | \(2 S V\) (kernel-accurate unary) | \(2 S V\) |
| 4 | `Multiply` (× cap) | \((S,V)\) | \(S V\) | \(S V\) |
| 5 | `Multiply` (× scale) | \((S,V)\) | \(S V\) | \(S V\) |
| 6 | `Exp` | \((S,V)\) | \(S V\) | \(S V\) |
| 7 | `ReduceSum` | \((S,V)\) | \(S V\) | \(S V\) |
| 8 | `Divide` | \((S,V)\) | \(S V\) | \(S V\) |
| 9 | CE tail | per token | \(\approx 2 S\) | \(\approx 2 S\) |
| **Identity total** | | | | **\(2 S d V + 7 S V + 2 S\)** |

(Steps 2–5 match the registered fused cap leaf total \(7 S V\); steps 6–8 match CE softmax leaf \(3 S V\).)

Full \((S,V)\) **raw** and **capped** logits may both be live briefly in eager PyTorch; Zepto identity chain materializes capped logits before `Exp`.

### 4.2 Fused region leaf (Liger + softcap, kernel-accurate)

| Step | Primitive | Effective tile | FLOPs | Subtotal |
|------|-----------|----------------|-------|----------|
| 1 | Chunked GEMM | streaming | \(2 S d V\) | \(2 S d V\) |
| 2 | Soft-cap on chunk | per element | \(7 S V\) | \(7 S V\) |
| 3 | Online vocab softmax + CE | per row | \(3 S V + 2 S\) | \(3 S V + 2 S\) |
| **Fused total** | | | | **\(2 S d V + 10 S V + 2 S\)** |

Fusion elides **resident full \((S,V)\)** logits; it does **not** remove cap or softmax **arithmetic**.

### 4.3 Paper-comparable (Appendix E style)

Paper LM-head often counts **GEMM only** (\(2 S d V\)) for logits. Adding cap + CE for training comparability:

\[
\mathrm{FLOPs}_{\mathrm{paper,train}} \approx 2 S d V \quad\text{(logits only)};\qquad
\mathrm{FLOPs}_{\mathrm{paper,full}} = 2 S d V + 10 S V + 2 S \ \text{(with cap + CE)}.
\]

**Closed form (kernel-accurate, default Zepto leaf):**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 2 S d V + 10 S V + 2 S
\]

**Delta vs uncapped `region/linear_ce`:** **\(+7 S V\)** forward (exactly the fused logit soft-cap leaf).

**Arithmetic intensity (numeric example):** Apertus-8B defaults \(S{=}8192\), \(d{=}4096\), \(V{=}131072\), bf16 \(e{=}2\):

| Quantity | Value |
|----------|-------|
| GEMM FLOPs | \(2 \times 8192 \times 4096 \times 131072 \approx 8.8 \times 10^{12}\) |
| Cap FLOPs | \(7 \times 8192 \times 131072 \approx 7.5 \times 10^9\) |
| Softmax+CE FLOPs | \(3 \times 8192 \times 131072 + 2 \times 8192 \approx 3.2 \times 10^9\) |
| Cap / GEMM ratio | \(\approx 0.09\%\) (cap is cheap vs GEMM at large \(V\)) |
| Min HBM (Liger peak logits tile, \(C{=}16\)) | \(\min(SVe,\; C S d e) =\) **1.0 GiB** (unchanged vs uncapped FLCE) |
| Vocab softmax AI (post-cap tile) | \(3/(2e) = 0.75\) FLOP/byte |

**Muse-scale note:** For \(V \sim 256\text{k}\), \(+7SV\) remains \(\ll 2SdV\) but is **required** for checkpoint FLOP parity vs matmul-only baselines in integration tests.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training with `requires_grad=True` on hidden and/or weight. Liger **recomputes** chunked logits (including softcap when enabled) in backward, analogous to uncapped FLCE.

| Step | Primitive | FLOPs | Subtotal |
|------|-----------|-------|----------|
| 1 | Recompute GEMM + cap forward on chunks | \(2 S d V + 7 S V\) | \(2 S d V + 7 S V\) |
| 2 | Softmax + CE VJP on capped logits | \(\approx 4 S V\) | \(4 S V\) |
| 3 | Cap VJP (tanh chain) | \(\approx 8 S V\) | \(8 S V\) |
| 4 | GEMM backward | \(\approx 4 S d V\) | \(4 S d V\) |
| **Backward total** | | | **\(\approx 6 S d V + 19 S V\)** |

Zepto registered partial recipes sum to **\(6 S d V + 7 S V + 8 S V = 6 S d V + 15 S V\)** when `region/linear_ce` backward leaf is composed with `region/logit_softcap` on the same tensor without merging recompute — parent `region/linear_ce_softcap` should use **one** recompute-aware leaf (\(\approx 6 S d V + 19 S V\)) to match Liger autograd.

`requires_grad=False` on hidden and weight → backward FLOPs = **0**.

**Closed form (training, fused recompute path — target leaf):**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} \approx 6 S d V + 19 S V
\]

**Closed form (composed partial regions today — avoid for default estimate):**
\[
\mathrm{FLOPs}_{\mathrm{bwd,partial}} \approx 6 S d V + 15 S V
\]

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|
| **Eager uncapped CE** | full logits \((S,V)\) | \(S V e\) | — |
| **Eager capped CE** | raw + capped logits \((S,V)\) each | up to **\(2 S V e\)** transient | — |
| **Liger FLCE + softcap** | one chunk \((\mathrm{chunk\_size}, V)\) | \(\min(SVe,\; C S d e)\), \(C{=}16\) | full \((S,V)\) |
| **Zepto identity (cap only region)** | capped output \((S,V)\) | \(S V e\) | on-chip tanh temps |

Cap does **not** change Liger chunk sizing vs uncapped FLCE; it adds on-chip \(\tanh\) temps only (no extra HBM `ALLOCATE` in fused leaf).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| **Eager capped** | logits or capped logits | \((S,V)\) | \(S V e\) | SAVE logits (bad) |
| **Liger + softcap** | hidden, weight | \((S,d)\), \((d,V)\) | \(S d e + V d e\) | SAVE hidden if grad; recompute capped chunks |
| **Liger + softcap** | capped logits | — | **0** | recompute in bwd |
| **Target fused leaf** | hidden | \((S,d)\) | \(S d e\) | `save_hidden: true` when grad |

### 6.3 Resource event chains

**Identity lowering (unfused `CappedFusedLinearCrossEntropy`):**
```
PERSIST(weight) → ALLOCATE(logits_raw[S,V]) → ALLOCATE(logits_capped[S,V]) → ALLOCATE(probs[S,V]) → SAVE(logits_capped) → ALLOCATE(loss_scalar)
```

**Fused region leaf (target `region/linear_ce_softcap/liger`):**
```
PERSIST(weight) → ALLOCATE(logits_chunk[chunk_size,V]) → ALLOCATE(loss_scalar) → SAVE(hidden) [if grad]
```

**Apertus-8B trace (fused, training, \(S{=}8192\), cap enabled):**
- Peak logits chunk: **1.0 GiB** (same as `docs/kernel/linear-ce.md`)
- Saved hidden: **64 MiB**
- Weight persistent: **≈ 1.0 GiB**
- Elided: full **2.0 GiB** \((S,V)\) logits; elide duplicate capped logits buffer

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF Muse/Gemma eager | Transformers | No | D | \(2SdV + 7SV + 3SV\) | materialized logits | full \((S,V)\) | identity |
| Liger FLCE + softcap | liger-kernel | Yes | D | \(2SdV + 10SV\) | recompute chunks + cap | \(\min(SVe, CSde)\) | `region/linear_ce_softcap/liger` (planned) |
| Liger FLCE | liger-kernel | Yes | D | \(2SdV + 3SV\) | recompute | \(\min(SVe, CSde)\) | `region/linear_ce/liger` |
| TRL Hub FLCE | trl-lib | Yes | D | \(2SdV + 3SV\) | vocab-tile | fp32 tile | `region/linear_ce/hub-trl` |
| Zepto partial | — | Partial | D | sum of identity ops | partial recipes | chunk if CE fused | `logit_softcap` + `linear_ce` |
| Zepto target | — | Yes | D | \(2SdV + 10SV\) | \(\approx 6SdV + 19SV\) | Liger peak | `region/linear_ce_softcap/*` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: linear_ce_softcap
recommended_region_ids:
  - id: region/linear_ce_softcap/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: D
    status: to_implement
  - id: region/linear_ce_softcap/liger
    variant: liger
    hardware_gate: exclude_xpu_mps
    fusion_boundary: D
    status: to_implement
  - id: region/linear_ce_softcap/hub-trl
    variant: hub-trl
    hardware_gate: cuda_only
    fusion_boundary: D
    status: to_implement
pattern_rule:
  op_families:
    - linear_matmul
    - divide
    - tanh
    - multiply
    - multiply
    - exp
    - reduce_sum
    - divide
    - gather
    - log
    - multiply
    - reduce_sum
recipe:
  forward_flops: "2*S*d*V + 10*S*V + 2*S"
  backward_flops: "6*S*d*V + 19*S*V (Liger recompute + cap VJP; use 6*S*d*V + 15*S*V only if composing separate linear_ce + logit_softcap leaves)"
  forward_flops_per_element_cap: 7
  backward_flops_per_element_cap: 8
  chunk_mem_const: 16
  materialize_full_logits: false
  save_logits: false
  save_hidden: true
  elided_temps: [logits_full, logits_capped_full, probs_full, tanh_intermediates_on_chip]
  saved_backward: [hidden]
  resource_events_forward:
    - PERSIST(weight)
    - ALLOCATE(logits_chunk)
    - ALLOCATE(loss_scalar)
    - SAVE(hidden)
  numerics_tags: [stable_fp32_online, vocab_axis, tanh_softcap]
capabilities: [fused]
priority: 7
variants:
  - reference
  - liger
  - hub-trl
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Liger-Kernel FLCE with optional softcap: [fused_linear_cross_entropy.py (ops)](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- Liger transformer wrapper: [fused_linear_cross_entropy.py (transformers)](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py)
- Dai et al., Liger-Kernel: [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
- Muse Glimmer logits + cap: [modeling_muse_glimmer.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py)
- Gemma 4 final logit softcap: [modeling_gemma4.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- Zepto uncapped CE research: [docs/kernel/linear-ce.md](../docs/kernel/linear-ce.md)
- Zepto authority: [docs/kernel-implementation.md §14](../docs/kernel-implementation.md)
- Step 7 plan (module + region intent): [docs/plans/step7-output-auxiliary-heads.md](../docs/plans/step7-output-auxiliary-heads.md)
- TRL Hub fused-linear-ce (uncapped baseline): [huggingface.co/trl-lib/fused-linear-ce](https://huggingface.co/trl-lib/fused-linear-ce)

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 7 | `region/linear_ce_softcap/liger` | §8 | D | Muse/Gemma training FLOPs + memory; +7SV fwd vs FLCE; Liger already supports `softcap=` in chunk loop |
| 7 | `region/linear_ce_softcap/reference` | §8 | D | Hardware-agnostic cost leaf for compose tests / non-Liger backends |
| 7 | `region/linear_ce_softcap/hub-trl` | §8 | D | CUDA Hub parity once cap semantics confirmed in TRL kernel |

**Open gaps:**
- **Parent region not registered:** `CappedFusedLinearCrossEntropy` does not match `region/linear_ce` pattern (12 ops vs 8); identity lowering bills all primitives — integration estimates rely on this until `region/linear_ce_softcap` lands.
- **Backward leaf reconciliation:** composed `linear_ce` + `logit_softcap` recipes under-count recompute vs Liger (\(15SV\) vs \(19SV\)) — architect should unify in one recipe dataclass.
- **TRL Hub:** no verified softcap API in Hub kernel; `hub-trl` variant may remain uncapped + separate cap unless TRL adds support.
- **Inference:** `LanguageModelOutput` + `region/logit_softcap` + `region/linear` — not this region (no CE).
- **MTP tails:** same cap policy on drafter heads — compose separate `CappedFusedLinearCrossEntropy` invocations per stage.
