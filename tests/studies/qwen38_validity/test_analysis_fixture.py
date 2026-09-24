"""Fixture CSV is analysis-ready and uses the locked H1/H2 columns."""

from __future__ import annotations

from pathlib import Path

from qwen38_validity import relative_error
from qwen38_validity.io import read_evaluation, read_ledger
from qwen38_validity.schema import EVALUATION_FIELDNAMES, LEDGER_FIELDNAMES

_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "studies"
    / "qwen38-validity"
    / "fixtures"
)


def test_fixture_evaluation_has_locked_columns() -> None:
    rows = read_evaluation(_FIXTURES / "evaluation.csv")
    assert len(rows) >= 8
    assert len({row.subject_id for row in rows}) >= 2
    for row in rows:
        assert row.target_vram != 0
        assert row.target_flop != 0
        re_vram = relative_error(row.y_vram, row.target_vram)
        re_flop = relative_error(row.y_flop_fcm, row.target_flop)
        assert abs(re_vram) < 0.10
        assert abs(re_flop) < 0.10
    header = (_FIXTURES / "evaluation.csv").read_text(encoding="utf-8").splitlines()[0]
    for name in EVALUATION_FIELDNAMES:
        assert name in header.split(",")
    assert "y_flop_fcm" in header
    assert "target_flop" in header


def test_fixture_ledger_is_readable() -> None:
    rows = read_ledger(_FIXTURES / "ledger.csv")
    assert len(rows) >= 1
    assert rows[0].failure_kind == "oom"
    header = (_FIXTURES / "ledger.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(LEDGER_FIELDNAMES)
