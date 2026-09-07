# kernel-architecture evals

Test cases are defined in `evals.json`. Use them to iterate on the skill in fresh agent sessions.

## Workspace layout

Pre-seed research from kernel-search eval outputs, then run architecture:

```
.pi/skills/kernel-search-workspace/iteration-1/eval-xielu-leaf/with_skill/outputs/research.md
  → copy to _workspace/research.md

.pi/skills/kernel-architecture-workspace/   # gitignored (create as needed)
└── iteration-1/
    ├── eval-xielu-from-research/
    │   ├── with_skill/outputs/architecture.md
    │   └── with_skill/grading.json
    └── eval-gqa-paged-from-research/
        └── ...
```

## Running an eval

1. Start a **fresh session**.
2. Copy prerequisite research to `_workspace/research.md` (see eval `prerequisite` in `evals.json`).
3. Activate `kernel-architecture` and run the eval `prompt`.
4. Save output to `with_skill/outputs/architecture.md`.
5. Validate:
   ```bash
   python3 .pi/skills/kernel-architecture/scripts/validate_architecture.py _workspace/architecture.md
   ```
6. Grade assertions manually or with an LLM; record evidence in `grading.json`.

## Iteration loop

1. Run all evals → grade → note failures.
2. Edit `SKILL.md`, `references/gotchas.md`, or the validator.
3. Re-run into `iteration-N+1/`.
4. Stop when pass rate plateaus and human review feedback is empty.

## Upstream dependency

Architecture evals require **completed kernel-search output**. If research fails validation, fix upstream first:

```bash
python3 .pi/skills/kernel-search/scripts/validate_research.py _workspace/research.md
```
