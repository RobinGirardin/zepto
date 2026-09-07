# Region pattern (rules + variants)

## Directory layout

```
implementations/regions/<slug>/
├── rules.py      # PatternMatchRule, ProvenanceMatchRule
├── variants.py   # Fused*RegionImplementation, *_REGIONS tuple
└── __init__.py   # re-exports
```

## rules.py

Two discovery paths (both usually defined; descriptor picks which applies):

1. **PatternMatchRule** — matches decomposed op chain from identity lowering
2. **ProvenanceMatchRule** — matches module invocation (`component_type`)

### Simple pattern (xIELU)

```python
XIELU_PATTERN = PatternMatchRule(
    id="pat-xielu-decomposed",
    kind="region/xielu",
    priority=5,
    op_families=("greater_than", "multiply", ...),
)

XIELU_PROVENANCE = ProvenanceMatchRule(
    id="prov-xielu",
    kind="region/xielu",
    priority=5,
    component_type="XIELU",
    require_contiguous_in_graph_order=True,
)
```

### Pattern with edge wiring (RMSNorm)

When op order alone is ambiguous, add `edge_constraints` and `constraints`:

```python
RMSNORM_PATTERN = PatternMatchRule(
    ...
    edge_constraints=(
        (0, 1, "output_to_input"),
        ...
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=1),
    ),
)
```

Map §8 `pattern_rule.op_families` directly; add edge constraints when research §6.3 identity chain has non-sequential tensor flow.

## variants.py structure

```python
@dataclass(frozen=True, slots=True)
class Fused{{Slug}}RegionImplementation:
    descriptor: RegionImplementationDescriptor
    recipe: {{Slug}}Recipe = DEFAULT_{{SLUG}}_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(self, region, graph, context) -> str | None:
        # 1. component_type check
        # 2. op family sequence vs descriptor.pattern_rule
        # 3. requested_capabilities (if gated)
        # 4. hardware_gate
        ...

    def lower(self, region, graph, context, edge_map, *, estimation, lowered_edges) -> LoweredNode:
        # ensure_lowered_edge for boundaries
        # build ResourceEvent list from recipe save_* flags
        # forward_flops / backward_flops from recipe
        ...


def _descriptor(*, impl_id: str, priority: int) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/{{slug}}",
        priority=priority,
        capabilities=frozenset({...}),
        provenance_rule={{SLUG}}_PROVENANCE,
        pattern_rule={{SLUG}}_PATTERN,
    )


FUSED_{{SLUG}}_REFERENCE = Fused{{Slug}}RegionImplementation(
    descriptor=_descriptor(impl_id="region/{{slug}}/reference", priority=5),
    ...
)

{{SLUG}}_REGIONS: tuple[Fused{{Slug}}RegionImplementation, ...] = (
    FUSED_{{SLUG}}_REFERENCE,
    ...
)
```

## Variant table mapping

| §8 field | variants.py |
|----------|-------------|
| `impl_id` | `RegionImplementationDescriptor.id` |
| `priority` | descriptor `priority` (higher wins on tie) |
| `hardware_gate` | `_check_hardware_gate()` in `compatible()` |
| `recipe` / saved policy | `recipe=` constructor arg |
| `capabilities` | descriptor `capabilities` frozenset |

## Multi-variant precedents

| Precedent | Variants | Gate style |
|-----------|----------|------------|
| xIELU | reference, cuda | `cuda_only` vs `any`; requires `fused` capability |
| RMSNorm | reference, liger, hub-xpu, hub-mps | `exclude_xpu_mps`, `xpu_only`, `mps_only` |

## lower() resource events

Translate §6.3 / §8 `resource_events_*` into:

- `ALLOCATE` — HBM-resident outputs and saved tensors
- `SAVE` — tensors kept for backward
- `RELEASE` — backward phase cleanup

**Omit** on-chip / SRAM temps listed in §8 `elided_temps`.
