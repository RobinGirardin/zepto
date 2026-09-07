# Architecture proposal: {{REGION_KIND}}

**Source:** _workspace/research.md (§8 YAML)
**Region kind:** region/{{slug}}
**Fusion boundary:** {{A|B|C|D}}
**Status:** to_implement

## Summary

[2–3 sentences: what gets fused, default variant, capability/hardware gates]

## New files

| Path | Purpose |
|------|---------|
| src/zepto/analysis/lowering/recipes/{{slug}}.py | Closed-form {{Slug}}Recipe dataclass |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/rules.py | PatternMatchRule + ProvenanceMatchRule |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/variants.py | Fused*RegionImplementation + descriptors |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/__init__.py | Export tuple |
| tests/lowering/regions/test_{{slug}}.py | Discovery, lowering, variant tests |
| src/zepto/modules/{{slug}}.py | **Only if** no reference module exists |

## Modified files

| Path | Change |
|------|--------|
| src/zepto/analysis/lowering/recipes/__init__.py | export {{Slug}}Recipe |
| src/zepto/analysis/lowering/implementations/regions/__init__.py | export {{SLUG}}_REGIONS constants |
| src/zepto/analysis/lowering/implementations/__init__.py | register in register_regions() |
| docs/kernel-implementation.md | mark region registered / add § cross-ref |

## Recipe design

[from §8 recipe block → dataclass fields]

```python
@dataclass(frozen=True, slots=True)
class {{Slug}}Recipe:
    forward_flops_per_element: int = ...
    backward_flops_per_element: int = ...
    # save_* / materialize_* flags from §8 saved_backward
```

## Discovery rules

[pattern_rule op_families → rules.py; edge_constraints if RMSNorm-style wiring matters]

```python
{{SLUG}}_PATTERN = PatternMatchRule(
    id="pat-{{slug}}-decomposed",
    kind="region/{{slug}}",
    op_families=(...),
)
{{SLUG}}_PROVENANCE = ProvenanceMatchRule(
    component_type="{{Component}}",
    ...
)
```

## Variants

| impl_id | hardware_gate | recipe | priority | notes |
|---------|---------------|--------|----------|-------|
| region/{{slug}}/reference | any | DEFAULT_{{SLUG}}_RECIPE | 5 | |
| region/{{slug}}/cuda | cuda_only | DEFAULT_{{SLUG}}_RECIPE | 8 | optional |

## Lowering behavior

[resource_events from §6.3 / §8 → lower() ALLOCATE/SAVE/RELEASE sequence]

- Forward: ...
- Backward: ...
- Elided on-chip temps (from §8 elided_temps): ...

## Tests

[Mirror test_xielu.py or test_rmsnorm.py]

- `test_compose_{{slug}}_discovers_region` — compose module, assert region kind and op count
- `test_{{slug}}_unfused_without_<capability>` — no fusion without requested capability
- `test_{{slug}}_fused_lowering_flops_and_allocs` — forward/backward FLOPs, ALLOCATE/SAVE counts
- `test_{{slug}}_fusion_map_absorbs_all_ops` — fusion_map covers identity chain
- `test_{{slug}}_<variant>_wins_on_<hardware>` — priority / hardware_gate routing

## Open questions / gaps

[G3, G4b, etc. from research §9 — design blockers for implementation PR]
