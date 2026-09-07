# AGENTS.md

## Harness: Zepto kernel pipeline

**Goal:** Research a kernel operation, produce an architecture proposal, then implement lowering + tests and archive research under `docs/kernel/`.

**Trigger:** Full kernel pipeline — run `/implement-kernel <operation>` (e.g. `/implement-kernel xIELU`). Simple questions can be answered directly without the pipeline.

**Artifacts:** `_workspace/research.md` (kernel-searcher) → `_workspace/architecture.md` (kernel-architect) → `src/` + `tests/` + `docs/kernel/<slug>.md` (kernel-implementer).

**Change log:**

| Date | Change | Target | Reason |
|------|--------|--------|--------|
| 2026-09-07 | Initial harness layout | agents, skills, prompts | pi-agent-harness pipeline |
| 2026-09-07 | Add kernel-implementer + 3-step chain | agents, prompts, AGENTS.md | Build from architecture; archive research with proposer |

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (RobinGirardin/zepto) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
