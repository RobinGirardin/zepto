# kernel-search evals

Test cases are defined in `evals.json`. Use them to iterate on the skill in fresh agent sessions.

## Workspace layout

Save eval outputs outside the skill directory (gitignored):

```
.pi/skills/kernel-search-workspace/
└── iteration-1/
    ├── eval-xielu-leaf/
    │   ├── with_skill/outputs/research.md
    │   ├── with_skill/grading.json
    │   └── without_skill/outputs/research.md
    ├── eval-masked-softmax-b/
    │   └── ...
    └── benchmark.json
```

## Running an eval

1. Start a **fresh session** (no leftover skill-editing context).
2. Activate `kernel-search` and run the eval `prompt` from `evals.json`.
3. Save output to `with_skill/outputs/research.md`.
4. Run baseline (no skill, or previous skill snapshot) → `without_skill/outputs/`.
5. Validate: `python3 .pi/skills/kernel-search/scripts/validate_research.py <path-to-research.md>`
6. Grade assertions manually or with an LLM; record evidence in `grading.json`.

## Iteration loop

1. Run all evals → grade → note failures.
2. Edit `SKILL.md`, `references/gotchas.md`, or the validator.
3. Snapshot previous skill if comparing versions.
4. Re-run into `iteration-N+1/`.
5. Stop when pass rate plateaus and human review feedback is empty.

See [agentskills.io evaluating skills](https://agentskills.io/skill-creation/evaluating-skills) for the full workflow.
