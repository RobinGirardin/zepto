#!/usr/bin/env python3
"""Validate a kernel-architecture proposal (_workspace/architecture.md)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REQUIRED_HEADINGS = [
    "## Summary",
    "## New files",
    "## Modified files",
    "## Recipe design",
    "## Variants",
    "## Tests",
]

OPTIONAL_HEADINGS = [
    "## Discovery rules",
    "## Lowering behavior",
    "## Open questions",
]

# Paths in markdown tables: backtick-quoted repo paths
PATH_IN_TABLE = re.compile(
    r"`((?:src|tests)/zepto/[^`]+)`|`((?:src|tests)/[^`]+)`"
)
REGION_KIND = re.compile(r"region/[a-z][a-z0-9_]*")
IMPL_ID = re.compile(r"region/[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)?")
YAML_FENCE = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ALREADY_REGISTERED = re.compile(
    r"\balready registered\b", re.IGNORECASE
)
MODULES_PATH = re.compile(
    r"(?:src/zepto/modules/|modules/)([a-z_0-9]+)\.py", re.IGNORECASE
)


def _repo_root(start: Path) -> Path:
    """Walk up from skill script to repo root."""
    for parent in [start, *start.parents]:
        if (parent / "src" / "zepto").is_dir():
            return parent
    return start


def check_headings(text: str) -> list[str]:
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"Missing required heading: {heading}")
    return errors


def check_path_conventions(text: str) -> list[str]:
    errors: list[str] = []
    for match in PATH_IN_TABLE.finditer(text):
        path = match.group(1) or match.group(2)
        if not path:
            continue
        if path.startswith("tests/") and "/lowering/regions/" not in path:
            if path.endswith(".py") and "test_" in path:
                errors.append(
                    f"Test path should be under tests/lowering/regions/: {path}"
                )
        if "implementations/regions/" in path:
            parts = path.split("implementations/regions/")
            if len(parts) > 1:
                slug_part = parts[1].split("/")[0]
                if slug_part and not re.fullmatch(r"[a-z][a-z0-9_]*", slug_part):
                    errors.append(f"Invalid region slug in path: {path}")
        if "/recipes/" in path and not path.endswith(".py"):
            errors.append(f"Recipe path should be a .py file: {path}")
    return errors


def extract_yaml_from_research(text: str) -> dict | None:
    match = YAML_FENCE.search(text)
    if not match:
        return None
    raw = match.group(1)
    if yaml is None:
        return {"_raw": raw} if "region_kind:" in raw else None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def check_research_crossref(arch_text: str, research_path: Path) -> list[str]:
    errors: list[str] = []
    if not research_path.is_file():
        return errors

    research_text = research_path.read_text(encoding="utf-8")
    yaml_data = extract_yaml_from_research(research_text)

    if yaml_data is None:
        errors.append(
            f"Research file missing parseable §8 YAML: {research_path}"
        )
        return errors

    region_kind = yaml_data.get("region_kind")
    if region_kind and str(region_kind) not in arch_text:
        errors.append(
            f"Architecture must mention region_kind from research: {region_kind}"
        )

    status = yaml_data.get("status")
    if status == "to_implement" and ALREADY_REGISTERED.search(arch_text):
        errors.append(
            'Research status is to_implement but architecture claims "already registered"'
        )

    recommended = yaml_data.get("recommended_region_ids") or yaml_data.get(
        "variants"
    )
    if isinstance(recommended, list):
        for item in recommended:
            impl_id = item.get("impl_id") if isinstance(item, dict) else item
            if impl_id and str(impl_id) not in arch_text:
                errors.append(
                    f"Architecture must mention recommended impl_id: {impl_id}"
                )
    elif isinstance(recommended, dict):
        for impl_id in recommended:
            if str(impl_id) not in arch_text:
                errors.append(
                    f"Architecture must mention recommended impl_id: {impl_id}"
                )

    return errors


def check_module_duplication(
    arch_text: str, repo_root: Path
) -> list[str]:
    warnings: list[str] = []
    modules_dir = repo_root / "src" / "zepto" / "modules"
    proposed: set[str] = set()
    for match in MODULES_PATH.finditer(arch_text):
        slug = match.group(1)
        if "Only if" in arch_text[max(0, match.start() - 80) : match.start()]:
            continue
        proposed.add(slug)

    for slug in proposed:
        candidates = [
            modules_dir / f"{slug}.py",
            modules_dir / f"{slug.replace('_', '')}.py",
        ]
        if slug == "rmsnorm":
            candidates.append(modules_dir / "rms_norm.py")
        for path in candidates:
            if path.is_file():
                warnings.append(
                    f"WARN: proposes modules/{slug}.py but {path.relative_to(repo_root)} already exists"
                )
                break
    return warnings


def validate(arch_path: Path, research_path: Path | None, repo_root: Path) -> list[str]:
    if not arch_path.is_file():
        return [f"File not found: {arch_path}"]

    text = arch_path.read_text(encoding="utf-8")
    issues: list[str] = []
    issues.extend(check_headings(text))
    issues.extend(check_path_conventions(text))

    if research_path is None:
        research_path = repo_root / "_workspace" / "research.md"
    issues.extend(check_research_crossref(text, research_path))
    issues.extend(check_module_duplication(text, repo_root))

    if not REGION_KIND.search(text) and "**Region kind:**" not in text:
        issues.append("Architecture should declare a region kind (region/<slug>)")

    return issues


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: {sys.argv[0]} <path-to-architecture.md> [path-to-research.md]",
            file=sys.stderr,
        )
        return 2

    script_dir = Path(__file__).resolve().parent
    repo_root = _repo_root(script_dir)

    arch_path = Path(sys.argv[1])
    research_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    issues = validate(arch_path, research_path, repo_root)
    errors = [i for i in issues if not i.startswith("WARN:")]
    warnings = [i for i in issues if i.startswith("WARN:")]

    for w in warnings:
        print(w, file=sys.stderr)

    if errors:
        print(f"FAIL: {arch_path}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"PASS: {arch_path}")
    if warnings:
        print(f"  ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
