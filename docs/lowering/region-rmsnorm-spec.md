# Region spec: `region/rmsnorm`

Implementation spec for fused RMSNorm lowering on Apertus. Pair with:

- [ADR-0009](../adr/0009-backend-profiles-and-attention-selection.md) — selection, presets
- [apertus-presets.md](./apertus-presets.md) — `apertus_hf_eager`, `apertus_hf_hub`
- [kernel-implementation.md](../kernel-implementation.md) §5 — FLOP/VRAM authority
- Template implementation: `src/zepto/analysis/lowering/implementations/regions/layernorm.py`

**Phase:** P1 (`apertus_hf_eager` partial). **Status:** planning — not implemented.

---

## 1. Purpose

Replace the 7-op eager RMSNorm decomposition with one lowered leaf that:

1. Bills **4·numel** forward FLOPs (Appendix E / Liger leaf), not ~5–6·numel from summing primitives.
2. **Elides** full-rank `squared` and `normalized` temporaries from the resource-event stream.
3. Saves **`rstd` only** (not `mean` — RMSNorm has no centering).
4. Works for **all** Apertus RMSNorm call sites: hidden pre-norms `(S, d)`, Q-norm `(h, S, d_h)`, K-norm `(h_kv, S, d_h)`, final norm.

Distinct from `region/layernorm` (mean + variance chain, 5·numel, saves `mean` + `inv_std`).

---

## 2. Structural source (`RMSNorm.forward`)

From `src/zepto/modules/rms_norm.py`:

```text
input x
  [0] multiply      x * x           → squared
  [1] reduce_sum    axis=last, keepdim
  [2] divide        / inv_norm_size → variance (mean of squares)
  [3] add           + eps
  [4] square_root
  [5] divide        x / denom       → normalized   (x is boundary input, fan-in)
  [6] parameter_scale  × γ (weight) → output      (Apertus: always on)
```

**Operation families (registry keys):**

| Step | Family | Notes |
|------|--------|-------|
| 0 | `multiply` | Self-multiply (both operands = `x`) |
| 1 | `reduce_sum` | Last axis, `keepdim=True` |
| 2 | `divide` | Divisor = `_inv_norm_size` graph input (boundary) |
| 3 | `add` | RHS = `_eps` graph input (boundary) |
| 4 | `square_root` | |
| 5 | `divide` | Numerator = `x` (boundary); denominator from step 4 |
| 6 | `parameter_scale` | γ parameter; skip when `elementwise_affine=False` |

**Module provenance:** `component_type="RMSNorm"` (`module_kind` on `RMSNorm`).

**QK-Norm:** `QKNormRMSNorm` delegates to two nested `RMSNorm` modules (`q_norm`, `k_norm`). Each inner `forward()` stamps **`component_type="RMSNorm"`** — one provenance envelope per norm, not the wrapper kind `QKNormRMSNorm`.

---

## 3. Discovery

### 3.1 Route

**Hybrid** (provenance envelope + internal pattern gate), same as LayerNorm:

```text
ProvenanceRegionMatcher(RMSNorm envelope)
  → PatternRegionMatcher.matches_subgraph(envelope.operation_ids, RMSNORM_PATTERN)
  → emit Region(kind="region/rmsnorm") or nothing
```

### 3.2 Provenance rule

```python
RMSNORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-rmsnorm",
    kind="region/rmsnorm",
    priority=5,
    component_type="RMSNorm",
    require_contiguous_in_graph_order=True,
)
```

### 3.3 Pattern rule (7-op affine — Apertus default)

```python
RMSNORM_PATTERN = PatternMatchRule(
    id="pat-rmsnorm-decomposed",
    kind="region/rmsnorm",
    priority=5,
    op_families=(
        "multiply",        # 0: square
        "reduce_sum",      # 1
        "divide",          # 2: mean of squares
        "add",             # 3: + eps
        "square_root",     # 4
        "divide",          # 5: x / rms
        "parameter_scale", # 6: × γ
    ),
    edge_constraints=(
        (0, 1, "output_to_input"),           # squared → reduce_sum
        (1, 2, "output_to_first_input"),     # sum → divide (left)
        (2, 3, "output_to_first_input"),     # variance → add (left)
        (3, 4, "output_to_input"),           # radicand → sqrt
        (4, 5, "output_to_second_input"),    # denom → divide (right)
        (5, 6, "output_to_input"),           # normalized → parameter_scale
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=1),
    ),
)
```

**Notes on pattern matching:**

- Step 0 self-multiply: both inputs are the same boundary edge `x`; matcher only validates inter-chain edges — no constraint that multiply is squaring (provenance + contiguous envelope is sufficient for v1).
- Step 5 numerator from `x`: fan-in from boundary, not from chain index — same pattern as LayerNorm step 5 (`centered` / external input).
- Two `divide` steps distinguished by **position** in `op_families`, not by family name alone.

### 3.4 No-affine variant (v2, optional)

6-op pattern ending at step 5 (`parameter_scale` omitted). Separate `PatternMatchRule` or `compatible()` rejects envelopes with 7 ops when variant expects 6. Not required for Apertus P1.

### 3.5 Negative cases (must not fuse)

| Case | Expected discovery |
|------|-------------------|
| Placeholder module (single `identity`) | Envelope fails pattern → no region |
| LayerNorm chain (has `subtract`) | Different families → no match |
| Manual inline norm without `RMSNorm` provenance | Pattern-only could match elsewhere — **hybrid prevents** unless inside envelope |

---

## 4. Region boundaries

Computed by `compute_region_boundaries()` on winning `operation_ids`.

| Boundary | Typical edges |
|----------|----------------|
| **Inputs** | `x`; optional `_inv_norm_size`, `_eps` (graph inputs, not parameters) |
| **Outputs** | Final output of step 6 (`parameter_scale`) |
| **Parameters** | `weight` (γ) on step 6 |

**Estimation port map** (for `RegionEstimationContext`):

| Key | Source |
|-----|--------|
| `input` | Primary boundary input (`x`) |
| `output` | Region output tensor |
| `weight` | γ from `parameter_tensors` |

---

## 5. Recipe

File: `src/zepto/analysis/lowering/recipes/rmsnorm.py` (new).

```python
@dataclass(frozen=True, slots=True)
class RMSNormRecipe:
    """Closed-form cost leaf for fused RMSNorm."""

    forward_flops_per_element: int = 4   # Appendix E
    backward_flops_per_element: int = 4  # v1: mirror forward when grad enabled

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel

    def rstd_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Shape of saved rstd: input with last dim replaced by 1."""
        if not input_shape:
            return (1,)
        return (*input_shape[:-1], 1)
```

**`reference` and `liger` variants share the same recipe in P1–P3** — Liger differs only in registration metadata (`capabilities`, future `backend` gate), not FLOP math.

### 5.1 Resource events (forward, training-aware)

Let `n = numel(x)`, `output` = lowered boundary output id, `rstd` = auxiliary id `region:{region.id}:rstd`.

| Event | Tensor | When |
|-------|--------|------|
| `ALLOCATE` | `output` | Always |
| `ALLOCATE` | `rstd` | Always (stats buffer for backward) |
| `SAVE` | `rstd` | Always in P1 forward; refine in P4 training preset |
| `SAVE` | `input` (`x`) | When `input.requires_grad` (P4 / training preset) |

**Elided (must NOT appear):** `squared`, intermediate `variance`, full-rank `normalized` — these are the VRAM win vs identity sum.

### 5.2 `LoweredNode` fields

| Field | Value |
|-------|-------|
| `id` | `f"region:{region.id}"` |
| `node_ids` | `region.operation_ids` (all 7) |
| `implementation` | `region/rmsnorm/reference` or `/liger` |
| `module_path` | `region.anchor.module_path` |
| `component_type` | `"RMSNorm"` |
| `region_id` | `region.id` |
| `forward_flops` | `4 * n` |
| `backward_flops` | `4 * n` if grad else `0` (P1 forward-only OK with `0`) |
| `auxiliary_edges` | `(rstd,)` |

---

## 6. Variants

File: `src/zepto/analysis/lowering/implementations/regions/rmsnorm/variants.py`.

| id | kind | priority | capabilities | `compatible()` extra |
|----|------|----------|--------------|----------------------|
| `region/rmsnorm/reference` | `region/rmsnorm` | 5 | `{"fused"}` | None |
| `region/rmsnorm/liger` | `region/rmsnorm` | 8 | `{"fused"}` | None (P3: optional `context.backend in ("hf+hub", "hf+flash2", "training+liger")`) |

**Preset selection (no pins):**

| Preset | Winning variant |
|--------|-----------------|
| `reference` | `reference` (priority sufficient when registered) |
| `apertus_hf_eager` | `reference` |
| `apertus_hf_hub` | `liger` (`requested_capabilities={"fused"}` + higher priority) |
| `apertus_hf_flash2` | `liger` |
| `apertus_training_liger` | `liger` |

**Do not reuse** `region/layernorm` descriptor or implementation class.

---

## 7. File layout

```text
src/zepto/analysis/lowering/
  recipes/
    rmsnorm.py                    # RMSNormRecipe
  implementations/regions/
    rmsnorm/
      __init__.py                 # export RMSNORM_REGIONS
      rules.py                    # RMSNORM_PATTERN, RMSNORM_PROVENANCE
      variants.py                 # FusedRMSNormRegionImplementation × 2
```

**Registration** in `implementations/__init__.py`:

```python
def register_regions(registry: LoweringRegistry) -> None:
    ...
    for impl in RMSNORM_REGIONS:
        registry.register_region(impl)
```

---

## 8. Apertus-8B call sites

Per decoder layer (+ final norm at model scope):

| Site | Module path (illustrative) | Input shape | numel (S=8192) |
|------|----------------------------|-------------|----------------|
| Pre-attention norm | `(..., "pre_attn_norm")` | `(S, d)` = `(8192, 4096)` | 33,554,432 |
| Q-norm | `(..., "attn", "qk_norm", "q_norm")` | `(h, S, d_h)` = `(32, 8192, 128)` | 33,554,432 |
| K-norm | `(..., "attn", "qk_norm", "k_norm")` | `(h_kv, S, d_h)` = `(8, 8192, 128)` | 8,388,608 |
| Pre-FFN norm | `(..., "pre_ffn_norm")` | `(S, d)` | 33,554,432 |
| Final norm | `("norm",)` or model root | `(S, d)` | 33,554,432 |

**Per-layer norm numel sum:** `2·S·d + S·d_h·(h + h_kv)` = `S·(8192 + 4096 + 1024)` = **13,312·S** → at S=8192: **≈ 109.0×10⁶** → forward FLOPs **≈ 436×10⁶** per layer (4×).

**QK-Norm Appendix E check:** `4·S·(h + h_kv)·d_h` = `4·8192·40·128` = **168 M** per layer — matches row sum for Q+K norms.

---

## 9. Tests

File: `tests/test_region_lowering.py` (extend) + optional `tests/lowering/regions/test_rmsnorm.py`.

### 9.1 Unit / graph fixtures

**`_build_rmsnorm_chain_graph()`** — mirror `_build_layernorm_chain_graph()`:

- Shapes: default `(4, 8)` for fast tests; parametrized `(8192, 4096)` and `(32, 8192, 128)` for Apertus shapes.
- Stamp `component_type="RMSNorm"`, `module_path=("Block", "pre_attn_norm")`.
- Wire 7 ops with families and edges from §3.3.
- Include `parameter_scale` with one γ parameter.

### 9.2 Test cases

| Test | Assert |
|------|--------|
| `test_discover_hybrid_rmsnorm_region` | Exactly one `region/rmsnorm`; 7 `operation_ids`; anchor `component_type=="RMSNorm"` |
| `test_discover_rmsnorm_rejects_layernorm_chain` | LayerNorm 6-op graph → zero `region/rmsnorm` |
| `test_build_plan_fuses_rmsnorm_block` | Plan has one `region` step consuming 7 ops |
| `test_lower_rmsnorm_elides_internal_allocs` | Lowered op count < 7; `fusion_map` maps all 7 structural ids → one lowered id |
| `test_rmsnorm_forward_flops_4n` | `forward_flops == 4 * numel(input)` |
| `test_rmsnorm_saves_rstd_not_mean` | `auxiliary_edges` contains `rstd`; no `mean` auxiliary |
| `test_rmsnorm_peak_vs_identity` | Sum of ALLOCATE bytes on fused path < identity path (exclude γ, params) |
| `test_apertus_block_discovers_four_rmsnorm_regions` | Compose `ApertusDecoderBlock` (8B dims, S=128 smoke); discover 4 regions/layer |
| `test_preset_hf_eager_selects_reference` | `apertus_hf_eager()` → `region/rmsnorm/reference` in selections |
| `test_preset_hf_hub_selects_liger` | `apertus_hf_hub()` → `region/rmsnorm/liger` (P3) |

### 9.3 Golden FLOPs (8B, S=8192, one pre_attn_norm)

```text
numel = 8192 * 4096 = 33_554_432
forward_flops = 4 * numel = 134_217_728
```

Compare to unfused identity sum: strictly greater (typically ~5–6× numel).

### 9.4 Golden VRAM order-of-magnitude (one hidden norm, bf16)

| Path | Dominant elidable temps |
|------|-------------------------|
| Identity (7 ops) | `squared` + `normalized` ≈ **2 × S × d × 2 B** ≈ **128 MiB** at S=8192, d=4096 |
| Fused `reference` | **0** full-rank temps; `rstd` ≈ **S × 1 × 2 B** ≈ **16 KiB** |

---

## 10. Implementation checklist

- [ ] `recipes/rmsnorm.py` — `RMSNormRecipe`
- [ ] `regions/rmsnorm/rules.py` — pattern + provenance
- [ ] `regions/rmsnorm/variants.py` — `reference` (+ `liger` stub with same recipe)
- [ ] Register in `register_regions()`
- [ ] `_build_rmsnorm_chain_graph()` test helper
- [ ] Discovery + plan + lower tests (§9.2)
- [ ] End-to-end: `ApertusDecoderBlock` compose → lower with `apertus_hf_eager` partial
- [ ] Docs: link from [apertus-presets.md](./apertus-presets.md) P1 row

---

## 11. Out of scope (this spec)

| Item | Track |
|------|-------|
| `region/fused_add_rmsnorm` (vLLM residual) | P5, separate kind |
| fp32 variance upcast byte model (HF eager numerics) | Optional `state`; not P1 |
| `elementwise_affine=False` 6-op pattern | v2 |
| Backward recipe refinement | P4 `apertus_training_liger` |
| `context.backend` filtering for `liger` | P3 ADR-0009 wiring |

---

## 12. Implementation order (within P1)

```text
1. rules.py + reference variant only
2. register + discovery tests
3. lower() + FLOP/VRAM tests
4. ApertusDecoderBlock integration test
5. liger variant (duplicate of reference, priority 8) — can land same PR or P3
```

---

*Spec derived from `rms_norm.py`, `layernorm.py` region template, ADR-0009, and kernel-implementation.md §5.*
