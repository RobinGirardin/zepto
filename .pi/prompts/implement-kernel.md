---
description: >
  Zepto kernel pipeline — research an operation then produce an architecture
  proposal. Use for "implement kernel X", "research and architect GQA/xIELU",
  "kernel pipeline", or when both research.md and architecture planning are needed.
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
    }
  ]
}
```

After the chain completes:

1. Confirm `_workspace/research.md` and `_workspace/architecture.md` exist.
2. Summarize: operation, region_kind, key files proposed, open gaps from §9.
3. Do **not** start implementation — the pipeline ends at the architecture proposal.

## Error handling

- If chain stops at step 1: report research failure; do not invoke kernel-architect manually without valid research.
- If step 2 fails but research exists: summarize partial architecture issues and point to `_workspace/` artifacts.

## Re-run / partial

- **Research only:** subagent single mode with `kernel-searcher` and the same task as step 1.
- **Architecture only** (research already in `_workspace/research.md`): subagent single mode with `kernel-architect`.
