"""Tests for ScatterUpdate batched indices."""

from __future__ import annotations

from zepto.compose import Tensor
from zepto.semantic import ScatterUpdate


def test_batched_rank2_indices() -> None:
    op = ScatterUpdate(axis=1)
    (output,) = op.infer_outputs(
        (
            Tensor(shape=(2, 16, 128)),
            Tensor(shape=(2, 4), semantic_type="placeholder_index"),
            Tensor(shape=(2, 4, 128)),
        )
    )
    assert output.shape == (2, 16, 128)
