---
description: >
  Full Zepto kernel pipeline in two subagent invocations — /architect-kernel then
  /implement-kernel. Use for "full kernel pipeline X", "research build and ship
  xIELU/GQA", without a single 3-step chain (avoids long bridged tool timeouts).
argument-hint: "<operation, e.g. xIELU or GQA paged decode>"
---

Run the **full** Zepto kernel pipeline for: **$@**

Use **two separate subagent tool calls** (never a single 3-step chain). Wait for each call to finish before starting the next.

---

## Phase 1 — Research + architecture

Use the **subagent** tool with `agentScope: "both"` and `confirmProjectAgents: false` in **chain** mode (two steps):

```json
{
  "agentScope": "both",
  "confirmProjectAgents": false,
  "chain": [
    {
      "agent": "kernel-searcher",
      "task": "Research $@ for Zepto cost modeling. Follow your agent workflow: load kernel-search skill, write the full report to _workspace/research.md, run validate_research.py until pass. Return HANDOFF block."
    },
    {
      "agent": "kernel-architect",
      "task": "Read _workspace/research.md (must exist and pass validation). Follow your agent workflow: load kernel-architecture skill, write the proposal to _workspace/architecture.md, run validate_architecture.py until pass. Prior step summary: {previous}. Return HANDOFF block."
    }
  ]
}
```

**If phase 1 fails:** report which step failed; do not run phase 2 without valid `_workspace/architecture.md`.

**If phase 1 succeeds:** confirm both workspace files exist and validators pass, then continue to phase 2.

---

## Phase 2 — Implementation

Prerequisite gate (run before delegating):

```bash
cd /Users/veoon/hslu/zepto-kernel-implementation
test -f _workspace/architecture.md && test -f _workspace/research.md
python3 .pi/skills/kernel-architecture/scripts/validate_architecture.py _workspace/architecture.md
```

Use the **subagent** tool with `agentScope: "both"` and `confirmProjectAgents: false` in **single** mode:

```json
{
  "agentScope": "both",
  "confirmProjectAgents": false,
  "agent": "kernel-implementer",
  "task": "Read _workspace/architecture.md and _workspace/research.md. Implement every file and registration edit in the architecture plan; run the planned pytest until green. Archive _workspace/research.md to docs/kernel/<slug>.md (kebab-case from §8 region_kind): copy full report and add **Proposer:** from `git config user.name` in the header block. Do not redesign architecture — report blockers in HANDOFF. Return HANDOFF block."
}
```

---

## Final summary

After both phases complete:

1. Confirm `_workspace/research.md`, `_workspace/architecture.md`, and `docs/kernel/<slug>.md` exist.
2. Summarize src/tests changed and pytest result from implementer HANDOFF.
3. Summarize: operation, `region_kind`, key files implemented, proposer on archived research, open gaps from research §9 / architecture.

## Error handling

| Failure | Action |
|---------|--------|
| Phase 1 step 1 (research) | Report only; no architect or implementer |
| Phase 1 step 2 (architecture) | Summarize research + architecture issues; no implementer |
| Phase 2 (implementation) | Summarize partial implementation, pytest, archive status; user can re-run `/implement-kernel` |

## Re-run / partial

- **Architect only:** `/architect-kernel $@`
- **Implement only:** `/implement-kernel` (when `_workspace/architecture.md` already valid)
