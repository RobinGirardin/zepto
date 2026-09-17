---
description: >
  Zepto kernel research + architecture — produces _workspace/research.md and
  _workspace/architecture.md. Use for "architect kernel X", "research and plan
  GQA/xIELU", or kernel design before implementation.
argument-hint: "<operation, e.g. xIELU or GQA paged decode>"
---

Run the Zepto kernel **research + architecture** pipeline for: **$@**

Use the **subagent** tool with `agentScope: "both"` and `confirmProjectAgents: false` in **chain** mode (two steps only — do not add implementer):

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

After the chain completes:

1. Confirm `_workspace/research.md` and `_workspace/architecture.md` exist.
2. Run validators if HANDOFF is ambiguous:

   ```bash
   cd /Users/veoon/hslu/zepto-kernel-implementation
   python3 .pi/skills/kernel-search/scripts/validate_research.py _workspace/research.md
   python3 .pi/skills/kernel-architecture/scripts/validate_architecture.py _workspace/architecture.md
   ```

3. Summarize: operation, `region_kind`, variants from §8, open gaps from research §9 / architecture.
4. Tell the user to run `/implement-kernel` when ready for code + tests + `docs/kernel/` archive.

## Error handling

- If chain stops at step 1: report research failure; do not invoke architect without valid research.
- If step 2 fails but research exists: summarize architecture issues; user may fix research and re-run architect only.

## Re-run / partial

- **Research only:** subagent single mode with `kernel-searcher` and the same task as step 1.
- **Architecture only** (research already in `_workspace/research.md`): subagent single mode with `kernel-architect` and the same task as step 2.
