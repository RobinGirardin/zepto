"""CSV and JSON artifacts for one study run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from apertus_validity.schema import (
    EVALUATION_FIELDNAMES,
    LEDGER_FIELDNAMES,
    SUBJECT_FIELDNAMES,
    EvaluationRow,
    LedgerRow,
    Subject,
    parse_evaluation_row,
    parse_ledger_row,
)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
    *,
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_subjects(path: Path, subjects: Iterable[Subject]) -> None:
    write_csv(path, SUBJECT_FIELDNAMES, (subject.to_csv_row() for subject in subjects))


def read_subjects(path: Path) -> list[Subject]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [Subject.from_csv_row(raw) for raw in csv.DictReader(handle)]


def write_evaluation(path: Path, rows: Iterable[EvaluationRow], *, append: bool = False) -> None:
    write_csv(
        path,
        EVALUATION_FIELDNAMES,
        (row.to_csv_row() for row in rows),
        append=append,
    )


def read_evaluation(path: Path) -> list[EvaluationRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [parse_evaluation_row(raw) for raw in csv.DictReader(handle)]


def write_ledger(path: Path, rows: Iterable[LedgerRow], *, append: bool = False) -> None:
    write_csv(
        path,
        LEDGER_FIELDNAMES,
        (row.to_csv_row() for row in rows),
        append=append,
    )


def read_ledger(path: Path) -> list[LedgerRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [parse_ledger_row(raw) for raw in csv.DictReader(handle)]


def write_run_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
