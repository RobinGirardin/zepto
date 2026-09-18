---
description: >
  Zepto kernel implementation from _workspace/architecture.md — recipes, regions,
  registration, tests, and docs/kernel/ archive. Use after /architect-kernel or
  when research + architecture already exist in _workspace/.
argument-hint: "[operation slug for logging, optional]"
---

Run the Zepto kernel **implementation** phase (operation **$@** if given — slug is optional; implementer reads §8 from `_workspace/research.md`).

## Prerequisite gate

Before delegating, verify architecture exists and validates:

```bash
cd /Users/veoon/hslu/zepto-kernel-implementation
test -f _workspace/architecture.md || { echo "Missing _workspace/architecture.md — run /architect-kernel first"; exit 1; }
test -f _workspace/research.md || { echo "Missing _workspace/research.md — run /architect-kernel first"; exit 1; }
python3 .pi/skills/kernel-architecture/scripts/validate_architecture.py _workspace/architecture.md
```

If the gate fails, stop and tell the user to run `/architect-kernel <operation>` first (or fix artifacts manually). Do not invoke implementer without valid architecture.

## Subagent (single mode — one call, no chain)

Use the **subagent** tool with `agentScope: "both"` and `confirmProjectAgents: false`:

```json
{
  "agentScope": "both",
  "confirmProjectAgents": false,
  "agent": "kernel-implementer",
  "task": "Read _workspace/architecture.md and _workspace/research.md. Implement every file and registration edit in the architecture plan; run the planned pytest until green. Archive _workspace/research.md to docs/kernel/<slug>.md (kebab-case from §8 region_kind): copy full report and add **Proposer:** from `git config user.name` in the header block. Do not redesign architecture — report blockers in HANDOFF. Return HANDOFF block."
}
```

After the subagent completes:

1. Confirm `docs/kernel/<slug>.md` exists (slug from research §8 `region_kind`, kebab-case).
2. Summarize src/tests files changed and pytest result from implementer HANDOFF.
3. Summarize: operation, `region_kind`, key files implemented, proposer name on archived research, open gaps from architecture not resolved.

## Error handling

- If implementer reports blockers: summarize partial changes, pytest failures, and whether research was archived.
- If subagent aborts or times out: report workspace state (`git status`, whether `docs/kernel/<slug>.md` exists) and suggest re-running `/implement-kernel` only.
