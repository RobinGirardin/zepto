---
status: accepted
---
# Backend profiles, attention backend, and region-kind boundaries

ADR-0002 defines context-driven implementation selection (pins, capabilities, priority) but leaves two costing axes unspecified: how a whole invocation names a **cost scenario** (`context.backend`, today unused in selection), and how **grouped-query attention** chooses among eager, SDPA, and FlashAttention-style region recipes. Region fusion taxonomy for cross-module kernels (fused QKV, fused add+RMSNorm) also needs a rule before Apertus parity work scales.

We adopt a layered selection model: **backend profile** filters scenario-appropriate variants; **attention backend** is a first-class invocation choice defaulting to `eager`; **implementation pins** remain exact overrides; **region kind** names one fusion category with shared discovery, not one directory per CUDA/Triton backend. Block-level fusion (`region/decoder_layer`) is out of scope for the foreseeable plan.

## Backend profile (`context.backend`)

**Role:** A human-readable **cost scenario label** for the whole invocation — e.g. `reference`, `hf+eager`, `hf+flash2`, `vllm`, `training+liger`. It is **not** a per-kernel pin and **not** executable backend code in Zepto.

**Selection (when wired):** After pins and before priority, implementations may declare scenario fit via `descriptor.requires` tags (e.g. `backend:vllm`) and/or `compatible()` checking `context.backend`. Presets set `backend` together with `requested_capabilities` and default priorities so users need not pin every region kind.

**Difference from pins:**

| Mechanism | Question it answers |
|-----------|---------------------|
| `context.backend` | "Which costing scenario am I modeling?" |
| `region_implementation_pins[kind]` | "For this fusion category, use exactly this implementation id." |
| `requested_capabilities` | "All chosen implementations must expose these tags." |
| `priority` | "Among compatible candidates, which default wins?" |

Pins **bypass** candidate filtering; backend profile **narrows** it. Golden tests and reproducibility use pins; everyday comparison ("cost like vLLM serve") uses presets + backend profile.

**Phasing:** `context.backend` exists today but selection ignores it until multiple complete backend profiles are registered. Early Apertus work may rely on capabilities + priority only; wiring `requires` / `compatible()` against `backend` lands when `reference`, `hf+eager`, and `vllm` (or similar) preset packs are stable.

## Attention backend

**Role:** Chooses the **cost leaf for grouped-query attention** — primarily VRAM shape (materialized `(h,S,S)` vs tile-linear) and softmax numerics policy — independent of unrelated kernels (RMSNorm hub, xIELU CUDA, etc.).

**Field:** Add `attention_backend: str` to `InvocationContext`, default **`"eager"`**. Values are costing profiles, not PyTorch dispatcher internals:

| Value | Modeled execution | Primary VRAM effect |
|-------|-------------------|---------------------|
| `eager` (default) | Decomposed matmul + mask + softmax + matmul | Peak `(h,S,S)` activations |
| `flash_attention_2` | `region/gqa` flash-style leaf | No full score/weight tensors in HBM |
| `sdpa` (later) | Explicit opt-in; separate registered variant under `region/gqa` | User-selected; not implied by default |

**Selection:** Maps to competing implementations sharing **`kind="region/gqa"`** (e.g. `region/gqa/eager`, `region/gqa/flash2`). Filter in `compatible()`:

```text
if context.attention_backend == "eager":
    reject flash/sdpa variants
elif context.attention_backend == "flash_attention_2":
    reject eager variant
```

Presets set `attention_backend` alongside `backend` (e.g. `apertus_hf_flash2`). Do **not** model SDPA dispatcher ambiguity unless the user sets `attention_backend="sdpa"`; if sub-dispatch matters later, add optional `sdpa_mode: "math" | "flash"` in `state` only for that case.

**Default policy:** Unspecified or `reference_invocation()` → `attention_backend="eager"`. Matches current Zepto decomposed GQA and HF `attn_implementation="eager"` baseline.

## Region kind for composite fusion

**`kind`** is the fusion **category** — registry key, discovery anchor, and pin key (e.g. `region/rmsnorm`, `region/gqa`, `region/fused_qkv`). **`id`** is the variant within that category (e.g. `region/rmsnorm/liger`).

**Composite kernels get a new `kind`, not an stretched provenance envelope:**

- Fused QKV (vLLM-style merged GEMM) → `region/fused_qkv` with its own pattern or module-scoped discovery; distinct from `region/linear`.
- Fused residual + RMSNorm → `region/fused_add_rmsnorm`; distinct from `region/rmsnorm` (different boundaries and saved tensors).

**Rejected:** Block-level `region/decoder_layer` fusing norm + attention + FFN in one leaf — too coarse for attribution, discovery, and incremental Apertus rollout.

## Selection precedence (full pass)

Extends ADR-0002 for regions:

```text
1. region_implementation_pins[kind]           → exact id, fail if incompatible
2. module_implementation_pins[component_type] → exact id, fail if incompatible
3. Filter: requested_capabilities ⊆ descriptor.capabilities
4. Filter: compatible() — includes attention_backend for kind=region/gqa;
           includes context.backend when backend profiles are wired
5. Sort by priority, stable tie-break on descriptor.id
6. Uncovered structural ops → per-op route (implementation_pins / family candidates)
```

Per-op `implementation_pins` unchanged. `allow_fallback` still governs whether an discovered-but-unlowerable region falls back to per-op steps.

## Presets

Preset packs live under `analysis/lowering/presets/` (in-tree until a second model family needs its own pack). Full Apertus scenario tables, variant matrices, and golden acceptance criteria: [`docs/lowering/apertus-presets.md`](../lowering/apertus-presets.md).

Presets set `backend`, `attention_backend`, and `requested_capabilities`; they do not pin every region kind. Pins remain for tests and explicit overrides.

## Considered options

- **Backend as duplicate pin map** — rejected; too verbose for whole-scenario costing.
- **SDPA as default or worst-case memory** — rejected; default `eager`; SDPA only when user opts in.
- **Stretch `region/linear` provenance for fused QKV** — rejected; provenance is per module invocation; cross-module fusion needs its own `kind`.
- **`attention_backend` only in untyped `state`** — rejected for the primary axis; first-class field with default; `state` reserved for secondary knobs (e.g. future `sdpa_mode`, fp32 softmax policy).

## Consequences

- Implement `attention_backend` on `InvocationContext` and `reference_invocation()` before registering `region/gqa` variants.
- First GQA region work ships `region/gqa/eager`; flash variant follows; both share one discovery rule and `kind`.
- `context.backend` filtering can land incrementally per preset pack without blocking RMSNorm/softmax regions.
- `docs/kernel-implementation.md` §11–§16 remain the FLOP/VRAM authority; this ADR governs **which** leaf is selected, not the leaf math.
- CONTEXT.md glossary gains **backend profile** and **attention backend**; implementation details stay in lowering code and kernel docs.

## Implementation order (planning)

1. ADR accepted; add `attention_backend` field (default `eager`).
2. `region/rmsnorm`, `region/softmax`, `region/masked_softmax` — backend-agnostic kinds, variants via capabilities.
3. `region/gqa/eager` + discovery; then `region/gqa/flash2` gated by `attention_backend`.
4. Preset module with `apertus_hf_eager` and `apertus_hf_flash2`.
5. Wire `context.backend` in `compatible()` as preset packs mature.
6. `region/fused_qkv`, `region/fused_add_rmsnorm` as separate kinds when vLLM parity requires them.
