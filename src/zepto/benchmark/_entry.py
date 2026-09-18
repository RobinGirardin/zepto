"""Console entry: bootstrap imports, then run CLI."""

from __future__ import annotations

from zepto.benchmark._paths import ensure_import_paths


def main() -> int:
    ensure_import_paths()
    from zepto.benchmark.cli import main as cli_main

    return cli_main()
