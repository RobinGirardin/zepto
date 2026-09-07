# Module pattern

When to propose `src/zepto/modules/<name>.py` vs region-only work.

## Default: region only

Most kernel-search targets already have (or will use) an existing nn.Module for compose tests:

| Module | component_type | Region |
|--------|----------------|--------|
| `modules/xielu.py` | `XIELU` | `region/xielu` |
| `modules/gqa.py` | `GQA` | `region/gqa` (planned) |
| `modules/softmax.py` | `Softmax` | — |
| `modules/rms_norm.py` | `RMSNorm` | `region/rmsnorm` |
| `modules/relu.py` | `ReLU` | relu region |

**Architecture rule:** if the module file exists, list it under "Existing dependencies" — not under "New files".

## When to propose a new module

Propose `src/zepto/modules/<slug>.py` only when **all** of:

1. No existing module emits the identity lowering chain
2. §8 `pattern_rule` must match a **module invocation** (`ProvenanceMatchRule.component_type`)
3. Tests need `compose_graph(lambda ctx: Module(), inputs)` and no substitute exists

## New module checklist

| Item | Detail |
|------|--------|
| Class | `nn.Module` subclass in `src/zepto/modules/<slug>.py` |
| Forward | Emit op chain matching §8 `pattern_rule.op_families` |
| Export | Add to `src/zepto/modules/__init__.py` if public |
| component_type | String matches `ProvenanceMatchRule.component_type` |
| Tests | `compose_graph` uses this module |

## Naming

- File: snake_case matching PyTorch convention (`rms_norm.py` not `rmsnorm.py`)
- Class: PascalCase (`RMSNorm`, `XIELU`)
- Slug for region paths may differ from module filename (`rmsnorm` region vs `rms_norm.py` module)

## GQA example

Research for GQA paged decode:

- `modules/gqa.py` **exists** → architecture proposes region + recipes only
- Flag **G3** (KV-cache invocation) as open question — module may need decode-path extension, not a duplicate file
