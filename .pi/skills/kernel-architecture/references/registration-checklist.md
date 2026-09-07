# Registration checklist

Every new region touches these registration sites. Missing any one = tests won't discover the implementation.

## 1. `recipes/__init__.py`

```python
from .{{slug}} import {{Slug}}Recipe

__all__ = [..., "{{Slug}}Recipe"]
```

## 2. `implementations/regions/<slug>/__init__.py`

```python
from .variants import (
    FUSED_{{SLUG}}_REFERENCE,
    {{SLUG}}_REGIONS,
)

__all__ = [
    "FUSED_{{SLUG}}_REFERENCE",
    "{{SLUG}}_REGIONS",
]
```

Export all public variant constants the tests or pins may reference.

## 3. `implementations/regions/__init__.py`

```python
from .{{slug}} import (
    FUSED_{{SLUG}}_REFERENCE,
    {{SLUG}}_REGIONS,
)

__all__ = [
    ...
    "FUSED_{{SLUG}}_REFERENCE",
    "{{SLUG}}_REGIONS",
]
```

## 4. `implementations/__init__.py` — `register_regions()`

```python
from .regions.{{slug}} import {{SLUG}}_REGIONS

def register_regions(registry: LoweringRegistry) -> None:
    ...
    for impl in {{SLUG}}_REGIONS:
        registry.register_region(impl)
```

For single-impl regions:

```python
registry.register_region(FUSED_{{SLUG}}_REGION)
```

## 5. `docs/kernel-implementation.md`

- Add or update the operation § with region ids and recipe reference
- Update registration table / backlog if present
- Cross-ref `_workspace/research.md` provenance if useful

## 6. Optional: `modules/__init__.py`

Only when proposing a **new** reference module.

## Verification command

After implementation (not architecture phase), tests should pass:

```bash
pytest tests/lowering/regions/test_{{slug}}.py -q
```

Architecture proposal should list all four Python registration files explicitly in "Modified files".
