---
name: kernel-architecture
description: >
  Turn Zepto kernel research (_workspace/research.md) into a full
  implementation architecture proposal: new files, registration changes,
  recipe dataclass, region rules/variants, optional reference module,
  and tests. Use after kernel-search or when the user asks for a kernel
  file plan, RegionImplementation design, or "how to implement" a region
  from YAML §8 (RMSNorm, Softmax, GQA, xIELU, RoPE, linear+CE, etc.).
compatibility: Requires _workspace/research.md from kernel-search (or equivalent).
metadata:
  version: "0.1"
  input: _workspace/research.md
  output: _workspace/architecture.md
  upstream_skill: kernel-search
---

# kernel-architecture

Turn **kernel-search output** into an actionable **implementation architecture proposal**.

**Trigger:** `/skill:kernel-architecture` or natural language (e.g. "propose file structure for GQA from research").

**Input:** `_workspace/research.md` (§8 YAML is the primary handoff artifact).

**Output:** write the complete proposal to `_workspace/architecture.md` using `assets/architecture-template.md`.

**Non-goals:** Do not write implementation code, tests, or registration edits — proposal only.

---

## Pipeline position

```
kernel-search → _workspace/research.md → kernel-architecture → _workspace/architecture.md → implementation PR
```

---

## Inputs (require or ask)

- **`_workspace/research.md`** — must exist and contain §8 YAML with `region_kind`, `pattern_rule`, variants, and `status`
- Extract from research:
  - **§8 YAML** — region ids, recipe fields, `elided_temps`, `save_*` flags, `status`
  - **§3** — fusion boundary (A/B/C/D), composition / mutual-exclusion rules
  - **§6.3** — identity and fused resource event chains
  - **§9** — open gaps (G1, G3, G4b, backlog rows)

If research is missing or §8 YAML is incomplete, stop and mention the user what information is missing.

---

## Authority read order (local code before proposing)

Read closest existing regions **before** drafting the file tree:

| Kernel shape | Closest precedent | Why |
|--------------|-------------------|-----|
| Simple activation (ReLU-style) | `regions/relu/` or `regions/xielu/` | Single variant, capability gate |
| Multi-variant + hardware routing | `regions/rmsnorm/` | Recipe-per-saved-tensor policy |
| Capability-gated fusion | `regions/xielu/` | `requested_capabilities` check |
| Attention / GQA | `regions/gqa/` (if present), `modules/gqa.py` | Existing module, paged variants |

Always read:

1. `src/zepto/analysis/lowering/implementations/regions/<closest>/` — `rules.py`, `variants.py`, `__init__.py`
2. `src/zepto/analysis/lowering/recipes/<closest>.py`
3. `tests/lowering/regions/test_<closest>.py`
4. `docs/kernel-implementation.md` — relevant § for the operation
5. `src/zepto/modules/` — grep for existing reference module (see decision tree)

Load detailed patterns on demand from `references/` (see table below).

---

## Decision tree

### Reference module exists?

```bash
ls src/zepto/modules/<slug>.py 2>/dev/null
grep -r "component_type=\"<Component>\"" src/zepto/analysis/lowering/implementations/regions/
```

- Module exists (e.g. `modules/xielu.py`, `modules/gqa.py`) → **do not** propose `modules/<slug>.py`
- No module and identity chain not decomposable from existing graph ops → propose `src/zepto/modules/<slug>.py`

See `references/module-pattern.md`.

### Multiple hardware / backend variants?

- One `*Recipe` dataclass per **saved-tensor policy** (RMSNorm pattern: reference vs liger vs hub-mps)
- Map §8 variant rows → `Fused*RegionImplementation` instances with `hardware_gate` and `priority`
- Export tuple: `SLUG_REGIONS = (REF, CUDA, …)`

See `references/region-pattern.md`.

### Capability gate?

- Document `requested_capabilities` in `compatible()` (xIELU `fused` pattern)
- Unfused path = identity lowering when capability absent; test both paths

### Already registered?

- If §8 `status: registered` → architecture describes **delta** (new variant, recipe tweak) not greenfield
- If `status: to_implement` → full file tree; do **not** claim "already registered"

---

## Workflow checklist

Copy and track:

```
- [ ] Read _workspace/research.md; extract §8 YAML, §3, §6.3, §9 gaps
- [ ] Read closest existing region + recipe + tests
- [ ] Grep modules/ for reference module; decide module touch
- [ ] Draft file tree (new + modified)
- [ ] Map §8 recipe block → *Recipe dataclass fields + forward_flops/backward_flops
- [ ] Map pattern_rule op_families → rules.py PatternMatchRule (+ edge_constraints if needed)
- [ ] Map variants → variants.py descriptors + hardware_gate + priority
- [ ] List registration edits (recipes/__init__, regions/__init__, register_regions)
- [ ] Plan tests mirroring test_<closest>.py
- [ ] Write _workspace/architecture.md from assets/architecture-template.md
- [ ] Run validate_architecture.py; fix failures; re-run until pass
```

---

## Output template

Fill `assets/architecture-template.md`. The proposal must be **actionable for the next agent/PR** — file paths, class names, registration hooks, test plan. Do **not** re-derive FLOP math (cite research §4/§5 instead).

---

## Reference files (load on demand)

| File | When to read |
|------|--------------|
| `references/file-touch-map.md` | Drafting new/modified file lists |
| `references/recipe-pattern.md` | Mapping §8 recipe → dataclass |
| `references/region-pattern.md` | rules.py + variants.py structure |
| `references/module-pattern.md` | Deciding whether to add `modules/` |
| `references/registration-checklist.md` | Every `__init__.py` touch point |
| `references/gotchas.md` | Before finishing; after failed eval |

---

## Quality checklist (self-verify before finishing)

- [ ] Every §8 `recommended_region_ids` entry appears in Variants table
- [ ] `region_kind` in proposal matches §8 YAML
- [ ] New files use repo path conventions (`recipes/`, `regions/<slug>/`, `tests/lowering/regions/`)
- [ ] Registration checklist complete (3 export sites + `register_regions`)
- [ ] Tests plan covers: compose → discover, unfused without capability, fused FLOPs/events, fusion_map
- [ ] Open questions from §9 carried forward (not silently dropped)
- [ ] No duplicate `modules/` when reference module already exists
- [ ] Explicit non-goals: no implementation code in this output
- [ ] `validate_architecture.py` passes

---

## Evals

Test cases live in `evals/evals.json`. Pre-seed `_workspace/research.md` from `kernel-search-workspace/.../outputs/research.md`. See `evals/README.md`.
