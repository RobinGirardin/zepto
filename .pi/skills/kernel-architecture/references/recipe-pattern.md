# Recipe pattern

Map §8 YAML `recipe` block → `src/zepto/analysis/lowering/recipes/<slug>.py`.

## Template

```python
"""Closed-form cost leaf for fused {{Slug}}."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class {{Slug}}Recipe:
    """Fused {{Slug}} costing shared by reference / backend variants.

    See docs/kernel-implementation.md §N and _workspace/research.md §4/§5.
    """

    forward_flops_per_element: int = ...
    backward_flops_per_element: int = ...
    # save_* / materialize_* from §8 saved_backward

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_{{SLUG}}_RECIPE = {{Slug}}Recipe()
```

## Field mapping from §8 YAML

| YAML key | Recipe field | Notes |
|----------|--------------|-------|
| `forward_flops_per_element` | `forward_flops_per_element` | Kernel-accurate leaf from research §4 |
| `backward_flops_per_element` | `backward_flops_per_element` | From §5; 0 when inference-only |
| `save_*` in `saved_backward` | `save_<tensor>: bool` | Drives ALLOCATE/SAVE in variants.py |
| `materialize_*` | `materialize_<tensor>: bool` | HBM-resident vs on-chip (RMSNorm `rstd`) |
| Variant-specific overrides | Named constants | `HUB_MPS_INFERENCE_RECIPE = RMSNormRecipe(save_rstd=False, ...)` |

## Precedents

**xIELU** — minimal flags, single default recipe:

```python
forward_flops_per_element: int = 8
backward_flops_per_element: int = 10
save_sign_mask: bool = True
```

**RMSNorm** — recipe-per-backend saved-tensor policy:

```python
materialize_rstd: bool = True
save_rstd: bool = True
# HUB_MPS_INFERENCE_RECIPE: save_rstd=False
```

## Rules

1. Recipes hold **closed-form constants only** — no graph or lowering logic.
2. `forward_flops` / `backward_flops` methods are the sole FLOP API for variants.
3. Shape helpers (e.g. `rstd_shape`) belong on recipe when saved tensor shape is non-obvious.
4. Cite research §4/§5 in docstring; do not re-derive in architecture doc — reference research.
