"""Relative error is (y - t) / t. Framework §2."""

from __future__ import annotations

import pytest

from apertus_validity import relative_error


def test_relative_error_positive_when_zepto_is_larger() -> None:
    assert relative_error(110, 100) == pytest.approx(0.10)


def test_relative_error_negative_when_zepto_is_smaller() -> None:
    assert relative_error(90, 100) == pytest.approx(-0.10)


def test_relative_error_zero_on_exact_agreement() -> None:
    assert relative_error(50, 50) == 0.0


def test_relative_error_rejects_zero_twin() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        relative_error(1, 0)
