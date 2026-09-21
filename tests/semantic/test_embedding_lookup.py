"""Tests for EmbeddingLookup batched token indices."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor
from zepto.semantic import EmbeddingLookup


def test_infer_outputs_rank1() -> None:
    op = EmbeddingLookup()
    (output,) = op.infer_outputs(
        (Tensor(shape=(12,), semantic_type="token_ids"),),
        parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
    )
    assert output.shape == (12, 64)


def test_infer_outputs_rank2_batched() -> None:
    op = EmbeddingLookup()
    (output,) = op.infer_outputs(
        (Tensor(shape=(4, 12), semantic_type="token_ids"),),
        parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
    )
    assert output.shape == (4, 12, 64)


def test_invalid_token_rank_raises() -> None:
    op = EmbeddingLookup()
    with pytest.raises(ValueError, match="rank-1 \\(S,\\) or rank-2"):
        op.infer_outputs(
            (Tensor(shape=(2, 4, 8), semantic_type="token_ids"),),
            parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
        )
