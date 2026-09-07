#!/usr/bin/env python3
"""Validate a kernel-search research report (_workspace/research.md)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REQUIRED_HEADINGS = [
    "## Section 0: Mathematical definition",
    "## Section 1: Three-layer costing model",
    "## Section 4: Forward FLOPs",
    "## Section 6: Memory",
    "## Section 8: Zepto implementation spec",
    "## Section 9: Sources",
]

OPTIONAL_HEADINGS = [
    "## Section 2:",
    "## Section 3:",
    "## Section 5: Backward FLOPs",
    "## Section 7: Ecosystem comparison",
]

URL_PATTERN = re.compile(r"https?://[^\s\)>\"]+")
YAML_FENCE = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
TBD_PATTERN = re.compile(r"\bTBD\b", re.IGNORECASE)


def check_headings(text: str) -> list[str]:
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"Missing required heading: {heading}")
    return errors


def check_bibliography_urls(text: str) -> list[str]:
    errors: list[str] = []
    section_9 = text.split("## Section 9:")[-1] if "## Section 9:" in text else ""
    if section_9 and not URL_PATTERN.search(section_9):
        errors.append("Section 9 bibliography must contain at least one https:// URL")
    return errors


def check_yaml_block(text: str) -> list[str]:
    errors: list[str] = []
    match = YAML_FENCE.search(text)
    if not match:
        errors.append("Missing fenced ```yaml block in Section 8")
        return errors

    raw = match.group(1)
    if yaml is None:
        if "region_kind:" not in raw:
            errors.append("YAML block must contain region_kind: (install PyYAML for full parse)")
        return errors

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        errors.append(f"YAML block does not parse: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append("YAML block must be a mapping (dict)")
        return errors

    if not data.get("region_kind"):
        errors.append("YAML block missing required field: region_kind")

    return errors


def check_tbd_in_flops(text: str) -> list[str]:
    """Warn if TBD appears in closed-form FLOP lines without a search plan nearby."""
    errors: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if TBD_PATTERN.search(line) and (
            "FLOPs" in line or "FLOP" in line or "\\mathrm{FLOPs}" in line
        ):
            context = "\n".join(text.splitlines()[max(0, i - 3) : i + 2])
            if "search plan" not in context.lower():
                errors.append(
                    f"Line {i}: TBD in FLOP line without nearby 'search plan'"
                )
    return errors


def validate(path: Path) -> list[str]:
    if not path.is_file():
        return [f"File not found: {path}"]

    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    errors.extend(check_headings(text))
    errors.extend(check_bibliography_urls(text))
    errors.extend(check_yaml_block(text))
    errors.extend(check_tbd_in_flops(text))
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-research.md>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    errors = validate(path)

    if errors:
        print(f"FAIL: {path}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"PASS: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
