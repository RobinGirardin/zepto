# Architecture gotchas

Mistakes specific to the research → architecture handoff. Read before finishing `_workspace/architecture.md`.

## Handoff

- **Do not re-derive FLOPs** — cite research §4/§5; architecture maps constants to recipe fields.
- **§8 YAML is authoritative** for `region_kind`, variant ids, and `status`. If prose in §0–7 conflicts with §8, follow §8 and note the conflict in Open questions.
- **`status: to_implement`** → full greenfield file plan; never write "already registered".
- **`status: registered`** → delta proposal only (new variant, recipe flag change).

## File planning

- **Duplicate modules** — proposing `modules/xielu.py` when `src/zepto/modules/xielu.py` exists is a common failure.
- **Slug vs filename** — region dir `rmsnorm` vs module file `rms_norm.py`; keep both in the proposal when they differ.
- **Missing registration** — listing new `variants.py` but forgetting `register_regions()` loop.
- **Wrong test path** — must be `tests/lowering/regions/test_<slug>.py`, not `tests/analysis/`.

## Region design

- **Capability gate omitted** — if research says fusion requires `requested_capabilities`, architecture must document the gate and unfused fallback test.
- **Hardware variants without priority** — every variant row needs priority; higher wins (cuda=8 over reference=5 in xIELU).
- **Edge constraints skipped** — when identity chain has fan-in/fan-out (RMSNorm), pattern-only matching is insufficient.
- **On-chip temps as ALLOCATE** — §8 `elided_temps` must not appear as HBM allocations in lowering plan.

## Variants vs recipes

- One recipe class with flags **when** variants differ only by save/materialize booleans.
- Separate recipe **instances** when FLOP constants or saved tensor shapes differ (RMSNorm hub-mps).

## Tests

- Mirror closest precedent (`test_xielu.py` for capability-gated; `test_rmsnorm.py` for multi-variant).
- Include **unfused** path test when capability-gated.
- Include **fusion_map** test asserting all identity ops absorbed.
- Pin test optional but document if variant selection is non-obvious.

## Open gaps

Carry forward from research §9 — do not hide blockers:

- **G1** — RoPE θ materialization
- **G3** — KV cache / decode invocation
- **G4/G4b** — unfused GQA false peak VRAM

Implementation PR may proceed with documented limitations; architecture must list them.

## Scope creep

- **No implementation code** in architecture output — pseudocode snippets in template sections are outlines only.
- **No kernel-search** — if research is missing, stop and run upstream skill.
