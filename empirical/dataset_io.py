"""Write empirical study artifacts to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from empirical.schema import (
    CONFIG_LINK_FIELDNAMES,
    EVALUATION_FIELDNAMES,
    OPTION_FIELDNAMES,
    ConfigurationLinkRow,
    EvaluationRow,
    OptionRow,
    write_csv_rows,
)


def write_dimension_tables(
    output_dir: Path,
    option_rows: Iterable[OptionRow],
    link_rows: Iterable[ConfigurationLinkRow],
) -> None:
    unique_options: dict[str, OptionRow] = {}
    for row in option_rows:
        unique_options[row.option_id] = row
    write_csv_rows(
        output_dir / "option_d.csv",
        OPTION_FIELDNAMES,
        (asdict(r) for r in unique_options.values()),
    )
    write_csv_rows(
        output_dir / "configuration_d.csv",
        CONFIG_LINK_FIELDNAMES,
        (asdict(r) for r in link_rows),
    )


def append_evaluation_rows(output_dir: Path, rows: Iterable[EvaluationRow]) -> None:
    write_csv_rows(
        output_dir / "evaluation.csv",
        EVALUATION_FIELDNAMES,
        (r.to_csv_row() for r in rows),
        append=True,
    )


def write_evaluation_rows(output_dir: Path, rows: Iterable[EvaluationRow]) -> None:
    write_csv_rows(
        output_dir / "evaluation.csv",
        EVALUATION_FIELDNAMES,
        (r.to_csv_row() for r in rows),
        append=False,
    )


def write_run_meta(output_dir: Path, meta: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_methodology(
    output_dir: Path,
    template_path: Path,
    run_parameters_md: str,
) -> None:
    body = template_path.read_text(encoding="utf-8")
    out = body.rstrip() + "\n\n## Run parameters\n\n" + run_parameters_md.strip() + "\n"
    (output_dir / "methodology.md").write_text(out, encoding="utf-8")
