"""Tests for lowering batch helpers."""

from __future__ import annotations

import pytest

from zepto.analysis.lowering.helpers import mlp_aux_shape, token_count, unpack_hidden
from zepto.compose import Tensor


def test_unpack_hidden_rank2_and_rank3() -> None:
    assert unpack_hidden(Tensor(shape=(8, 16))) == (1, 8, 16)
    assert unpack_hidden(Tensor(shape=(2, 8, 16))) == (2, 8, 16)


def test_token_count() -> None:
    assert token_count(Tensor(shape=(8, 16))) == 8
    assert token_count(Tensor(shape=(2, 8, 16))) == 16


def test_mlp_aux_shape_matches_boundary_rank() -> None:
    assert mlp_aux_shape(Tensor(shape=(8, 16)), 32) == (8, 32)
    assert mlp_aux_shape(Tensor(shape=(2, 8, 16)), 32) == (2, 8, 32)


def test_unpack_hidden_rejects_other_ranks() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        unpack_hidden(Tensor(shape=(8,)))
    with pytest.raises(ValueError, match="rank-2"):
        unpack_hidden(Tensor(shape=(2, 8, 16, 4)))
