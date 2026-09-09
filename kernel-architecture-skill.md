Here is a concrete skill-structure exploration for the **research → architecture proposal** stage, aligned with [AgentSkills.io](https://agentskills.io/home), [Pi skills](https://pi.dev/docs/latest/skills), and your existing `kernel-search` skill.

---

## Pipeline: two skills, one handoff

```mermaid
flowchart LR
  A["/skill:kernel-search"] --> B["_workspace/research.md"]
  B --> C["/skill:kernel-architecture"]
  C --> D["_workspace/architecture.md"]
  D --> E["Implementation PR"]
```

| Stage | Skill | Input | Output |
|-------|-------|-------|--------|
| Research | `kernel-search` | operation name, scope | `_workspace/research.md` (§0–9 + YAML §8) |
| Architecture | `kernel-architecture` (new) | `_workspace/research.md` | `_workspace/architecture.md` (file plan + design) |
| Build | normal agent work | architecture doc | code + tests |

The handoff artifact is **Section 8 YAML** in research — it already encodes `region_kind`, `pattern_rule`, `recipe`, variants, and `status: to_implement`. The architecture skill turns that machine-readable spec into a repo-faithful file plan.

---

## What AgentSkills + Pi require

Both standards agree on the same shape your `kernel-search` already follows:

| Principle | Spec guidance | Your `kernel-search` | Apply to `kernel-architecture` |
|-----------|---------------|----------------------|--------------------------------|
| **Progressive disclosure** | Metadata (~100 tok) → SKILL.md (<5k tok) → references on demand | `references/*.md`, `assets/report-template.md` | Same: keep SKILL.md thin; put file maps & examples in `references/` |
| **Description = trigger** | WHAT + WHEN, keywords, ≤1024 chars | Lists ops, RegionImplementation, docs path | List "architecture proposal", "file plan", "from research.md", recipe/region paths |
| **Coherent unit** | One skill = one workflow | Research only | Architecture only — **not** implementation |
| **Templates** | Output format in `assets/` | `assets/report-template.md` | `assets/architecture-template.md` |
| **Validation loops** | Scripts + self-checklist | `scripts/validate_research.py` | `scripts/validate_architecture.py` (checks file list, YAML cross-ref) |
| **Evals** | Train/validation queries for triggering | `evals/evals.json` + workspace | Mirror with architecture prompts |

Pi-specific notes:
- Location: `.pi/skills/kernel-architecture/` (project-scoped, trusted)
- Invocation: `/skill:kernel-architecture` or auto-trigger from description
- `name` need not match directory in Pi (but matching is cleaner)

You already have an empty stub at `.pi/skills/kernel-architecture/SKILL.md` — good starting point.

---

## Proposed directory layout

Mirror `kernel-search` so the pair feels like a pipeline:

```
.pi/skills/kernel-architecture/
├── SKILL.md                          # Core workflow (~150–250 lines)
├── assets/
│   └── architecture-template.md      # Output skeleton
├── references/
│   ├── file-touch-map.md             # Every file type + registration hooks
│   ├── recipe-pattern.md             # recipes/<kernel>.py conventions
│   ├── region-pattern.md             # rules.py + variants.py conventions
│   ├── module-pattern.md             # When src/zepto/modules/ is needed
│   ├── registration-checklist.md     # __init__.py touch points
│   └── gotchas.md                    # Architecture-specific mistakes
├── scripts/
│   └── validate_architecture.py      # Structural checks on output
└── evals/
    ├── evals.json
    └── README.md
```

**Progressive disclosure rule:** SKILL.md tells the agent *when* to read each reference file; don't inline the full xIELU/RMSNorm walkthrough.

---

## SKILL.md skeleton

```yaml
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
```

### Body sections (recommended)

1. **Inputs** — require `_workspace/research.md`; extract §8 YAML, §3 boundaries, §6 resource chains
2. **Authority read order** — local code before proposing:
   - Closest existing region (`xielu` for activations, `rmsnorm` for multi-variant, `relu` for simple)
   - `src/zepto/analysis/lowering/recipes/`
   - `tests/lowering/regions/test_<kernel>.py`
   - `docs/kernel-implementation.md` relevant §
3. **Decision tree** — conditional branches (AgentSkills "conditional workflow"):
   - Reference module exists? → skip `modules/`
   - Multiple hardware variants? → one recipe dataclass per saved-tensor policy (RMSNorm pattern)
   - Capability gate? → document `requested_capabilities` (xIELU `fused` pattern)
4. **Workflow checklist** — plan-validate-execute pattern:
   - Parse §8 YAML
   - Draft file tree
   - Map recipe fields → `*Recipe` dataclass
   - Map pattern_rule → `rules.py` `PatternMatchRule`
   - Map variants → `variants.py` descriptors
   - List registration edits
   - List tests
   - Write `_workspace/architecture.md` from template
   - Run validator
5. **Output template pointer** — `assets/architecture-template.md`
6. **Quality checklist** — self-verify before finishing
7. **Explicit non-goals** — do not write implementation code; proposal only

---

## Output template (`assets/architecture-template.md`)

The architecture doc should be **actionable for the next agent/PR**, not a repeat of research math:

```markdown
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
| src/zepto/analysis/lowering/recipes/{{slug}}.py | Closed-form XIELURecipe-style dataclass |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/rules.py | PatternMatchRule + ProvenanceMatchRule |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/variants.py | Fused*RegionImplementation + descriptors |
| src/zepto/analysis/lowering/implementations/regions/{{slug}}/__init__.py | Export tuple |
| tests/lowering/regions/test_{{slug}}.py | Discovery, lowering, variant tests |
| src/zepto/modules/{{slug}}.py | **Only if** no reference module exists |

## Modified files
| Path | Change |
|------|--------|
| recipes/__init__.py | export {{Slug}}Recipe |
| implementations/regions/__init__.py | export {{SLUG}}_REGIONS |
| implementations/__init__.py | register in register_regions() |
| docs/kernel-implementation.md | mark region registered / add § cross-ref |

## Recipe design
[from §8 recipe block → dataclass fields]

## Discovery rules
[pattern_rule op_families → rules.py, with edge_constraints if needed]

## Variants
| impl_id | hardware_gate | recipe | priority | notes |
|---------|---------------|--------|----------|-------|

## Lowering behavior
[resource_events from §6.3 / §8 → lower() ALLOCATE/SAVE/RELEASE sequence]

## Tests
[Mirror test_xielu.py: discover, unfused without capability, fused flops, fusion_map]

## Open questions / gaps
[G3, G4b, etc. from research §9]
```

---

## File touch map (from existing kernels)

Your repo has a repeatable pattern. The architecture skill should encode this in `references/file-touch-map.md`:

### Always (new region)

| Artifact | Example | Notes |
|----------|---------|-------|
| Recipe | `recipes/xielu.py` | `@dataclass` with `forward_flops()`, `backward_flops()` |
| Rules | `regions/xielu/rules.py` | `PatternMatchRule` + `ProvenanceMatchRule` |
| Variants | `regions/xielu/variants.py` | `Fused*RegionImplementation`, `_descriptor()`, export tuple |
| Package init | `regions/xielu/__init__.py` | Re-export implementations |
| Tests | `tests/lowering/regions/test_xielu.py` | compose → discover → lower → assert FLOPs/events |

### Registration (always edit)

```99:107:src/zepto/analysis/lowering/implementations/__init__.py
def register_regions(registry: LoweringRegistry) -> None:
    """Register fused region implementations."""
    registry.register_region(RELU_REGION)
    registry.register_region(FUSED_LINEAR_REGION)
    registry.register_region(FUSED_LAYERNORM_REGION)
    for impl in RMSNORM_REGIONS:
        registry.register_region(impl)
    for impl in XIELU_REGIONS:
        registry.register_region(impl)
```

Also: `recipes/__init__.py`, `regions/__init__.py`.

### Conditional: reference module

Only when research says identity lowering has **no** `src/zepto/modules/<kernel>.py`:

- GQA already has `modules/gqa.py` → architecture proposes **region only**
- A new `region/softmax` might need `modules/softmax.py` if not decomposable from existing graph ops

Decision rule for the skill: **grep `src/zepto/modules/` and `component_type` in provenance rule before proposing a new module.**

---

## Calibrating control (AgentSkills best practice)

| Part of architecture skill | Specificity | Why |
|-----------------------------|-------------|-----|
| File paths & registration hooks | **Low** (prescriptive) | Consistency across kernels |
| Variant count / hardware gates | **Medium** | Follow §8 YAML + RMSNorm precedent |
| Test assertions | **Medium** | Template from `test_xielu.py` |
| Implementation details inside `lower()` | **High** (outline only) | Research already has resource chains; code agent decides edge cases |

---

## Description optimization (trigger evals)

Draft description (train set ideas):

**Should trigger:**
- "I have research in `_workspace/research.md` — propose the file structure for GQA paged attention"
- "Architecture plan for region/masked_softmax from the YAML spec"
- "What files do I need to add for xIELU lowering?" (after research exists)
- "Turn §8 into an implementation checklist"

**Should NOT trigger:**
- "Research FlashAttention for Zepto" → `kernel-search`
- "Implement the xIELU region now" → normal coding (architecture already done)
- "Review this PR" → unrelated

Near-miss negatives are important: queries mentioning "kernel" or "region" but asking for research vs implementation vs architecture.

---

## Validation script (recommended)

Lightweight checks `validate_architecture.py` can enforce:

1. Output file exists at `_workspace/architecture.md`
2. Contains sections: New files, Modified files, Recipe design, Variants, Tests
3. Every listed path under `src/` or `tests/` uses repo conventions
4. §8 `region_kind` / `recommended_region_ids` appear in the proposal
5. If research says `status: to_implement`, architecture must not claim "already registered"
6. Warn if proposing `modules/` when `component_type` module already exists

This mirrors `kernel-search`'s validator-driven finish gate.

---

## Eval strategy (parallel to kernel-search)

```json
{
  "skill_name": "kernel-architecture",
  "evals": [
    {
      "id": "xielu-from-research",
      "prompt": "Using _workspace/research.md from xIELU research, write the architecture proposal.",
      "assertions": [
        "Lists recipes/xielu.py and regions/xielu/{rules,variants}.py",
        "Documents requested_capabilities fused gate",
        "Includes test_xielu.py plan",
        "Does not propose new modules/xielu.py (already exists)"
      ]
    },
    {
      "id": "gqa-paged-from-research",
      "prompt": "Architecture proposal for GQA paged decode from existing research.",
      "assertions": [
        "Proposes region/gqa/paged variants from §8 YAML",
        "Notes G3 KV-cache gap",
        "References modules/gqa.py for identity chain, no duplicate module"
      ]
    }
  ]
}
```

Run evals **with research.md pre-seeded** in the workspace (copy from `kernel-search-workspace/.../outputs/research.md`).

---

## How this differs from `kernel-search`

| | kernel-search | kernel-architecture |
|--|---------------|---------------------|
| Question answered | *What* does the kernel cost? | *Where* does code go? |
| Primary authority | Papers, HF source, docs | Existing region implementations |
| Output depth | Math, FLOPs, VRAM, YAML spec | File tree, classes, registration, tests |
| Token budget | Heavy (web search) | Light (local codebase reads) |
| §8 YAML | **Produces** | **Consumes** |

Keep them separate — combining would violate AgentSkills "coherent unit" guidance and bloat triggering.

---

## Suggested next steps

1. **Fill `SKILL.md`** using the skeleton above (~200 lines max)
2. **Add `assets/architecture-template.md`** — the highest-leverage single file
3. **Extract `references/region-pattern.md`** from `xielu` + `rmsnorm` (your best dual example: simple vs multi-variant)
4. **Write `validate_architecture.py`** — even a 50-line script unlocks eval loops
5. **Run one manual end-to-end test:** GQA research already in `_workspace/research.md` → `/skill:kernel-architecture` → review proposal quality
6. **Iterate description** with ~20 trigger eval queries per [optimizing descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)

---

## Quick sanity check on current research

Your `_workspace/research.md` (GQA paged decode) is ready for this skill: §8 YAML defines three `region/gqa/paged` variants, `pattern_rule` op families, recipe closed forms, and `elided_temps`. An architecture proposal would **not** re-derive FLOPs — it would map that YAML to `recipes/gqa.py`, `regions/gqa/{rules,variants}.py`, registration, and tests, while flagging **G3** (no decode/KV invocation yet) as an open design item.

If you want, I can draft the full `kernel-architecture/SKILL.md` + template + validator in a follow-up turn.
