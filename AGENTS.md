# AGENTS.md

## Harness: Zepto kernel pipeline

**Goal:** Research a kernel operation, then produce an implementation architecture proposal (file plan + design).

**Trigger:** Kernel research + architecture for Zepto lowering — run `/implement-kernel <operation>` (e.g. `/implement-kernel xIELU`). Simple questions can be answered directly without the pipeline.

**Artifacts:** `_workspace/research.md` (kernel-searcher) → `_workspace/architecture.md` (kernel-architect).

**Change log:**

| Date | Change | Target | Reason |
|------|--------|--------|--------|
| 2026-09-07 | Initial harness layout | agents, skills, prompts | pi-agent-harness pipeline |

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (RobinGirardin/zepto) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
