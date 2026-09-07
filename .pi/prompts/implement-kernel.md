---
description: >
  Zepto kernel pipeline — research, architecture proposal, then implementation.
  Use for "implement kernel X", "research and build GQA/xIELU", "kernel pipeline",
  or full lowering from operation name through code and docs/kernel/ archive.
argument-hint: "<operation, e.g. xIELU or GQA paged decode>"
---

Run the Zepto kernel pipeline for: **$@**

Use the **subagent** tool with `agentScope: "both"` in **chain** mode:

```json
{
  "agentScope": "both",
  "chain": [
    {
      "agent": "kernel-searcher",
      "task": "Research $@ for Zepto cost modeling. Follow your agent workflow: load kernel-search skill, write the full report to _workspace/research.md, run validate_research.py until pass. Return HANDOFF block."
    },
    {
      "agent": "kernel-architect",
      "task": "Read _workspace/research.md (must exist and pass validation). Follow your agent workflow: load kernel-architecture skill, write the proposal to _workspace/architecture.md, run validate_architecture.py until pass. Prior step summary: {previous}. Return HANDOFF block."
    },
    {
      "agent": "kernel-implementer",
      "task": "Read _workspace/architecture.md and _workspace/research.md. Implement every file and registration edit in the architecture plan; run the planned pytest until green. Archive _workspace/research.md to docs/kernel/<slug>.md (kebab-case from §8 region_kind): copy full report and add **Proposer:** from `git config user.name` in the header block. Do not redesign architecture — report blockers in HANDOFF. Prior step summary: {previous}. Return HANDOFF block."
    }
  ]
}
```

After the chain completes:

1. Confirm `_workspace/research.md`, `_workspace/architecture.md`, and `docs/kernel/<slug>.md` exist.
2. Confirm implementation: summarize src/tests files changed and pytest result from implementer HANDOFF.
3. Summarize: operation, region_kind, key files implemented, proposer name on archived research, open gaps from research §9 / architecture.

## Error handling

- If chain stops at step 1: report research failure; do not invoke later agents without valid research.
- If step 2 fails but research exists: summarize architecture issues; do not run implementer without valid architecture.
- If step 3 fails but architecture exists: summarize partial implementation, pytest failures, and whether research was archived.

## Re-run / partial

- **Research only:** subagent single mode with `kernel-searcher` and the same task as step 1.
- **Architecture only** (research already in `_workspace/research.md`): subagent single mode with `kernel-architect`.
- **Implementation only** (architecture already in `_workspace/architecture.md`): subagent single mode with `kernel-implementer` and the same task as step 3.
