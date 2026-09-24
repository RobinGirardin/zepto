"""CSV / JSON io: Qwen subject fieldnames plus shared evaluation / ledger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from qwen38_validity.schema import SUBJECT_FIELDNAMES, Subject
from validity_common.io import (
    read_evaluation,
    read_ledger,
    write_csv,
    write_evaluation,
    write_ledger,
    write_run_meta,
)


def write_subjects(path: Path, subjects: Iterable[Subject]) -> None:
    write_csv(path, SUBJECT_FIELDNAMES, (subject.to_csv_row() for subject in subjects))


def read_subjects(path: Path) -> list[Subject]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [Subject.from_csv_row(raw) for raw in csv.DictReader(handle)]


__all__ = [
    "read_evaluation",
    "read_ledger",
    "read_subjects",
    "write_csv",
    "write_evaluation",
    "write_ledger",
    "write_run_meta",
    "write_subjects",
]
