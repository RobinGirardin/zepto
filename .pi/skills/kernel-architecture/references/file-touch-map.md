# File touch map

Repeatable artifact layout for a **new** Zepto fused region. Use as a checklist when drafting `_workspace/architecture.md`.

## Always (new region)

| Artifact | Path pattern | Notes |
|----------|--------------|-------|
| Recipe | `src/zepto/analysis/lowering/recipes/<slug>.py` | `@dataclass` with `forward_flops()`, `backward_flops()` |
| Rules | `src/zepto/analysis/lowering/implementations/regions/<slug>/rules.py` | `PatternMatchRule` + `ProvenanceMatchRule` |
| Variants | `src/zepto/analysis/lowering/implementations/regions/<slug>/variants.py` | `Fused*RegionImplementation`, `_descriptor()`, export tuple |
| Package init | `src/zepto/analysis/lowering/implementations/regions/<slug>/__init__.py` | Re-export implementations + `*_REGIONS` tuple |
| Tests | `tests/lowering/regions/test_<slug>.py` | compose → discover → lower → assert FLOPs/events |

### Slug conventions

- Directory / file slug: lowercase, underscores (`rmsnorm`, `xielu`, `gqa`)
- Recipe class: PascalCase + `Recipe` (`XIELURecipe`)
- Region export tuple: `UPPER_SLUG_REGIONS` (`XIELU_REGIONS`)
- Region kind string: `region/<slug>` (`region/xielu`)

## Registration (always edit)

| File | Change |
|------|--------|
| `recipes/__init__.py` | Import and add to `__all__` |
| `implementations/regions/__init__.py` | Import constants + add to `__all__` |
| `implementations/__init__.py` | `register_regions()`: `for impl in SLUG_REGIONS: registry.register_region(impl)` |

Example from `register_regions()`:

```python
for impl in XIELU_REGIONS:
    registry.register_region(impl)
```

Single-variant regions may register one constant (`RELU_REGION`) instead of a tuple.

## Conditional: reference module

| Condition | Action |
|-----------|--------|
| `src/zepto/modules/<slug>.py` exists | Region only — identity chain comes from module |
| No module, decomposable from existing ops | Region only — pattern matches op chain |
| No module, needs new nn.Module for compose tests | Add `src/zepto/modules/<slug>.py` + export in `modules/__init__.py` |

**Decision rule:** grep `src/zepto/modules/` and `component_type` in provenance rule before proposing a new module.

Existing modules (do not duplicate): `xielu.py`, `gqa.py`, `softmax.py`, `rms_norm.py`, `relu.py`, `rope.py`, …

## Conditional: docs

| Condition | Action |
|-----------|--------|
| `status: to_implement` in §8 | Add backlog row or § stub in `docs/kernel-implementation.md` |
| New fusion boundary or variant family | Cross-ref from relevant § |

## Conditional: recipe variants

When §8 lists multiple implementations with different saved-tensor policies:

- Prefer **one recipe dataclass** with flags (`save_rstd`, `materialize_rstd`) over duplicate classes when policies differ only by booleans
- Use **separate recipe instances** (e.g. `HUB_MPS_INFERENCE_RECIPE`) when FLOP constants or shapes differ materially (RMSNorm pattern)
