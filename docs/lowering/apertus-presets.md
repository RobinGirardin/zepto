# Apertus lowering presets (planning)

Cost-scenario bundles for Apertus-8B / Apertus-70B lowering. Each preset returns a fully specified `InvocationContext` (see [ADR-0009](../adr/0009-backend-profiles-and-attention-selection.md)).

**Authority:** Kernel FLOPs and VRAM leaves — [`kernel-implementation.md`](../kernel-implementation.md). HF / vLLM wiring and gap tracker — [`apertus-transformers-implementation.md`](../../apertus-transformers-implementation.md).

**Not implemented yet.** This document is the acceptance checklist for preset functions under `src/zepto/analysis/lowering/presets/apertus.py`.

---

## Backend string vocabulary (`hf`, `hub`, …)

Preset names and `context.backend` values use **`+`-separated tags**. Each tag names a **cost scenario** Zepto should model — not executable code Zepto runs. The structural graph is identical across presets; only **which region variant** wins at lowering time changes (FLOP leaves, elided temps, SAVE events).

| Tag | What it refers to (real world) | What Zepto selects differently |
|-----|--------------------------------|--------------------------------|
| **`hf`** | [HuggingFace Transformers](https://github.com/huggingface/transformers) Apertus stack — modular/generated `modeling_apertus.py`, shared Llama kernels | Baseline target for parity; pairs with an attention mode (`eager`, `flash2`, …) |
| **`hub`** | Transformers **`USE_HUB_KERNELS=YES`** — hub-downloaded fused kernels ([`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)) | Fused **elementwise/reduction** variants: `region/rmsnorm/liger`, optional `region/rope_apply/hub`. Does **not** change attention by itself |
| **`eager`** | HF `attn_implementation="eager"` — explicit matmul + fp32 softmax + matmul | `attention_backend="eager"` → `region/gqa/eager`; materialized `(h,S,S)`; static causal mask |
| **`flash2`** | HF `attn_implementation="flash_attention_2"` (Dao FlashAttention family) | `attention_backend="flash_attention_2"` → `region/gqa/flash2`; no full score/weight tensors; no `S×S` mask alloc |
| **`training+liger`** | Liger-style fused kernels during **training** (forward + backward SAVE policy) | Same fused norm/activation leaves as hub, plus backward FLOPs and explicit SAVE events |
| **`vllm`** | [vLLM](https://github.com/vllm-project/vllm) Apertus serve path — fused QKV, fused add+RMSNorm, FA attention | vLLM-specific **kinds**: `region/fused_qkv`, `region/fused_add_rmsnorm`; flash GQA + serve layout |
| **`reference`** | Zepto internal baseline | Identity per-op lowering; minimal fusion (existing proof regions only) |

**`hf` vs `hub` in one sentence:** `hf` says *which library stack* you are comparing against; `hub` says *that stack has hub-kernel substitutions enabled* for norms/RoPE (and similar), which switch Zepto from decomposed per-op costing to fused region leaves that elide intermediate VRAM.

**Concrete hub hooks for Apertus** (from [`apertus-transformers-implementation.md`](../../apertus-transformers-implementation.md) §5.1):

| HF hook | Hub package / kernel | Zepto region variant |
|---------|----------------------|----------------------|
| `@use_kernel_forward_from_hub("RMSNorm")` on `ApertusRMSNorm` | `kernels-community/liger-kernels` → `LigerRMSNorm` | `region/rmsnorm/liger` — 4·numel FLOPs, save `rstd` only, elide `squared`/`normalized` |
| `@use_kernel_forward_from_hub("rotary_pos_emb")` | `kernels-community/rotary` | `region/rope_apply/hub` — elide rotate-half temps |
| Attention | *not hub-accelerated* | Controlled by **`attention_backend`**, not `hub` |
| xIELU | optional external CUDA wheel (not in hub mapping) | `region/xielu/cuda` when modeling nickjbrowning/XIELU |

**Example backend strings:**

```text
hf+eager     → Transformers Apertus, eager attention, no hub kernels
               (Python RMSNorm, decomposed RoPE, matmul+softmax GQA)

hf+hub       → Transformers + USE_HUB_KERNELS, attention still eager
               (Liger RMSNorm, hub rotary; GQA still materializes S×S)

hf+flash2    → Transformers + flash_attention_2 + typically hub norms/rope
               (Liger RMSNorm, FA-2 GQA, no S×S mask buffer)
```

Zepto never imports or runs HF, Liger, or FlashAttention. Presets only ensure lowered **FLOPs and ResourceEvents** match what those stacks would allocate if you ran them on CUDA with the same config.

---

## Preset summary

| Preset | Models | Reference stack | Primary use |
|--------|--------|-----------------|-------------|
| `reference` | Any | Per-op identity + existing regions | Regression baseline, debugging |
| `apertus_hf_eager` | Apertus | HF `attn_implementation="eager"`, no hub | Atto eager golden, Appendix E parity |
| `apertus_hf_hub` | Apertus | HF eager attn + `USE_HUB_KERNELS` (Liger RMSNorm, rotary) | HF hub-kernel inference costing |
| `apertus_hf_flash2` | Apertus | HF flash attn + hub norms/rope | HF production CUDA inference |
| `apertus_training_liger` | Apertus | Training forward/backward, Liger-style fused norms/FFN activations | Training VRAM/FLOP estimates |
| `apertus_vllm_serve` | Apertus | vLLM production serve path | Serve-time peak VRAM vs Zepto |

---

## Per-preset `InvocationContext`

Shared defaults unless noted: `phase="forward"`, `hardware="cuda"`, `default_dtype=BF16`, `allow_fallback=True`.

### `reference()`

Zepto baseline — no scenario assumptions beyond identity lowering.

| Field | Value |
|-------|-------|
| `backend` | `"reference"` |
| `attention_backend` | `"eager"` |
| `requested_capabilities` | `∅` |
| Pins | none |

**Region variants (implicit via priority):**

| Kind | Variant id | Notes |
|------|------------|-------|
| `region/relu` | `region/relu` | Pattern-only |
| `region/linear` | `region/linear` | Provenance-only; same FLOPs as identity |
| `region/layernorm` | `region/layernorm` | Non-Apertus |
| All other ops | `{family}/identity` | Decomposed |

**When to use:** CI regression, before/after fusion diffs, structural graph inspection.

---

### `apertus_hf_eager()`

Matches HuggingFace Apertus with **`attn_implementation="eager"`** and **no hub kernels**.

| Field | Value |
|-------|-------|
| `backend` | `"hf+eager"` |
| `attention_backend` | `"eager"` |
| `requested_capabilities` | `∅` (fusion optional; unfused acceptable) |
| `state` | `("softmax_numerics", "stable_fp32")` — optional tag for G5; does not insert cast ops by default |

**Target region variants:**

| Kind | Variant id | Priority | Status |
|------|------------|----------|--------|
| `region/rmsnorm` | `region/rmsnorm/reference` | 5 | Planned (G4a) |
| `region/softmax` | — | — | N/A (inside GQA) |
| `region/masked_softmax` | `region/masked_softmax/reference` | 5 | Planned (G4b) |
| `region/gqa` | `region/gqa/eager` | 5 | Planned (G4d) |
| `region/linear` | `region/linear` | 5 | ✓ Registered |
| `region/xielu` | — (identity chain) | 0 | Optional; unfused OK for v1 |
| Causal mask | per-op `materialized_causal_mask` | — | ✓ Static `(1,S,S)` |
| RoPE apply | per-op decomposed | — | ✓ No `region/rope_apply` in v1 |
| RoPE init | host setup (G1) | — | Planned |

**VRAM signature (8B, S=8192, one layer, dominant terms):**

- Mask: shared `(1,S,S)` ≈ 128 MiB bf16
- GQA eager: up to 2–3× `(h,S,S)` ≈ 8–12 GiB peak accounting if unfused
- RMSNorm: 2× full `(S,d)` temps per pre-norm site if unfused

**Golden target:** Atto / Appendix E FLOP script at S ∈ {8192, 65536}.

---

### `apertus_hf_hub()`

HF with **`USE_HUB_KERNELS=YES`** — Liger RMSNorm, rotary hub; attention still eager unless flash preset used.

| Field | Value |
|-------|-------|
| `backend` | `"hf+hub"` |
| `attention_backend` | `"eager"` |
| `requested_capabilities` | `{"fused"}` |

**Target region variants:**

| Kind | Variant id | `requires` / capabilities |
|------|------------|----------------------------|
| `region/rmsnorm` | `region/rmsnorm/liger` | `fused`, `backend:hf+hub` |
| `region/rope_apply` | `region/rope_apply/hub` | `fused` (optional v2) |
| `region/gqa` | `region/gqa/eager` | — |
| `region/xielu` | identity or `region/xielu/reference` | unfused OK |
| `region/linear` | `region/linear` | unchanged |

**VRAM delta vs `apertus_hf_eager`:** Elide RMSNorm `squared` / `normalized` temps; elide RoPE rotate-half when `region/rope_apply` lands.

---

### `apertus_hf_flash2()`

HF **`attn_implementation="flash_attention_2"`** + hub norms/rope (typical production CUDA inference).

| Field | Value |
|-------|-------|
| `backend` | `"hf+flash2"` |
| `attention_backend` | `"flash_attention_2"` |
| `requested_capabilities` | `{"fused", "flash_attention"}` |

**Target region variants:**

| Kind | Variant id | Selection gate |
|------|------------|----------------|
| `region/gqa` | `region/gqa/flash2` | `attention_backend == "flash_attention_2"` |
| `region/rmsnorm` | `region/rmsnorm/liger` | capabilities |
| `region/masked_softmax` | absorbed into `region/gqa/flash2` | no standalone alloc |
| Causal mask | none (causal inside FA tile) | no `(S,S)` buffer |
| `region/rope_apply` | `region/rope_apply/hub` | optional |

**VRAM signature (8B, S=8192):** Drop per-layer `(h,S,S)` ≈ 4+ GiB vs eager; drop shared mask at S=65536 saves ≈ 8 GiB vs materialized.

**Not modeled in v1:** `flex_attention`, flash-attn v3/v4 naming — map to `flash_attention_2` leaf until distinct recipes exist.

---

### `apertus_training_liger()`

Training forward/backward with Liger-style fused elementwise/reduction kernels (not vLLM serve layout).

| Field | Value |
|-------|-------|
| `backend` | `"training+liger"` |
| `attention_backend` | `"eager"` (FA training optional later) |
| `phase` | `"forward"` or `"backward"` via argument |
| `requested_capabilities` | `{"fused", "training"}` |

**Target region variants:**

| Kind | Variant id | Training notes |
|------|------------|----------------|
| `region/rmsnorm` | `region/rmsnorm/liger` | SAVE `rstd`; backward FLOPs from recipe |
| `region/xielu` | `region/xielu/reference` | Optional CUDA variant same FLOP leaf |
| `region/gqa` | `region/gqa/eager` | Full `(h,S,S)` unless FA training preset added |
| `region/linear` | `region/linear` | SAVE activations for backward |

**Optional later:** `attention_backend="flash_attention_2"` + FA backward recompute policy (save `(m,ℓ)` not `P`).

---

### `apertus_vllm_serve()`

vLLM [`model_executor/models/apertus.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py) production inference.

| Field | Value |
|-------|-------|
| `backend` | `"vllm"` |
| `attention_backend` | `"flash_attention_2"` |
| `requested_capabilities` | `{"fused", "flash_attention", "tensor_parallel"}` |
| `state` | `("tensor_parallel", False)` default — TP changes GEMM shapes, not single-GPU peak template |

**Target region variants (vLLM-specific kinds):**

| Kind | Variant id | Replaces |
|------|------------|----------|
| `region/fused_qkv` | `region/fused_qkv/vllm` | three × `region/linear` for Q/K/V |
| `region/rmsnorm` | `region/rmsnorm/vllm` or `liger` | same FLOP leaf |
| `region/fused_add_rmsnorm` | `region/fused_add_rmsnorm/vllm` | separate add + `region/rmsnorm` |
| `region/gqa` | `region/gqa/flash2` | vLLM `Attention` backend |
| `region/xielu` | `region/xielu/cuda` | optional; Python fallback same FLOPs |
| `region/linear` | column/row parallel leaves | Phase 2; single-GPU preset may keep unfused GEMMs |

**VRAM delta vs `apertus_hf_flash2`:** Fused add+RMSNorm elides residual temp; fused QKV elides duplicate reads of hidden states (traffic, not extra tensors).

---

## Region kind → variant matrix

Rows = kinds; columns = presets. Cell = winning **variant id** when registered and compatible.

| Kind | reference | hf_eager | hf_hub | hf_flash2 | training_liger | vllm |
|------|-----------|----------|--------|-----------|----------------|------|
| `region/rmsnorm` | identity* | reference | liger | liger | liger | vllm / liger |
| `region/gqa` | identity* | eager | eager | flash2 | eager | flash2 |
| `region/masked_softmax` | identity* | reference | reference | (in gqa) | reference | (in gqa) |
| `region/linear` | linear | linear | linear | linear | linear | linear |
| `region/fused_qkv` | — | — | — | — | — | vllm |
| `region/fused_add_rmsnorm` | — | — | — | — | — | vllm |
| `region/xielu` | identity | identity | identity | identity | reference | cuda? |
| `region/rope_apply` | identity | identity | hub | hub | identity | hub |
| `region/relu` | relu | relu | relu | relu | relu | relu |

\*Until region registered, per-op identity chain.

---

## Capabilities vocabulary

| Capability | Meaning |
|------------|---------|
| `fused` | Fused kernel leaf; elides internal temps |
| `flash_attention` | FlashAttention-style GQA VRAM model |
| `training` | Backward FLOPs + SAVE events enabled |
| `tensor_parallel` | TP-aware GEMM variants (future) |
| `residual_fusion` | Fused add+norm (`region/fused_add_rmsnorm`) |
| `stable_fp32_softmax` | fp32 softmax tile byte model (G5) |

Implementations declare `capabilities`; presets set `requested_capabilities` as a hard filter (ADR-0009).

---

## Acceptance criteria (golden tests)

For each preset, when all planned regions are registered:

1. **FLOPs:** Forward total within tolerance of Atto / Appendix E script for counted ops (RMSNorm, QK-Norm, softmax, MLP GEMMs).
2. **Peak VRAM:** Order-of-magnitude match at S ∈ {8192, 65536} for dominant terms in gap doc (mask, `(h,S,S)`, RMSNorm temps).
3. **Lowering record:** Every fused region has `fusion_map` entries; no double-lowering of absorbed ops.
4. **Selection audit:** `region_selections` explain preset-driven choices without manual pins.

Suggested test matrix:

| Test | Preset | S | Assert |
|------|--------|---|--------|
| `test_apertus_8b_eager_flops` | `apertus_hf_eager` | 8192 | FLOPs vs golden file |
| `test_apertus_8b_eager_peak_vram` | `apertus_hf_eager` | 8192 | Mask + scores present |
| `test_apertus_8b_flash_no_scores` | `apertus_hf_flash2` | 8192 | No `(h,S,S)` ALLOCATE in GQA |
| `test_apertus_8b_flash_long_ctx_mask` | `apertus_hf_flash2` | 65536 | No `S²` mask buffer |
| `test_preset_selects_liger_rmsnorm` | `apertus_hf_hub` | 8192 | `region/rmsnorm/liger` chosen |

---

## Implementation phasing

| Phase | Presets shippable | Regions required |
|-------|-------------------|------------------|
| **P0** (now) | `reference` only | existing: linear, layernorm, relu |
| **P1** | `apertus_hf_eager` partial | + rmsnorm/reference, masked_softmax/reference |
| **P2** | `apertus_hf_eager` full | + gqa/eager, G1 RoPE init |
| **P3** | `apertus_hf_hub`, `apertus_hf_flash2` | + rmsnorm/liger, gqa/flash2 |
| **P4** | `apertus_training_liger` | backward recipes on P1–P3 regions |
| **P5** | `apertus_vllm_serve` | + fused_qkv, fused_add_rmsnorm |

`context.backend` filtering in `compatible()` wires at **P3** when at least two backend profiles have distinct variant sets.

---

## API sketch

```python
# src/zepto/analysis/lowering/presets/apertus.py (future)

def reference(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...

def apertus_hf_eager(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...

def apertus_hf_hub(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...

def apertus_hf_flash2(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...

def apertus_training_liger(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...

def apertus_vllm_serve(*, phase: str = "forward", dtype: DType = DType.BF16) -> InvocationContext: ...
```

All delegate to `reference_invocation()` with preset-specific fields from the tables above.

---

## Related decisions

- [ADR-0002](../adr/0002-context-driven-lowering-selection.md) — selection and pins
- [ADR-0009](../adr/0009-backend-profiles-and-attention-selection.md) — backend profile, attention backend, region kind boundaries
