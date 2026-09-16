"""Tests for Equal semantic operation."""

from __future__ import annotations

import pytest

from zepto.analysis.resolved import ResolvedValue
from zepto.compose import Tensor
from zepto.semantic import Equal, EstimationContext
from zepto.semantic.metadata import DType


def _resolved(tensor: Tensor) -> ResolvedValue:
    return ResolvedValue(tensor=tensor, role=None, dtype=tensor.dtype or DType.FP32)


def test_broadcast_mask() -> None:
    (mask,) = Equal().infer_outputs(
        (
            Tensor(shape=(4, 2)),
            Tensor(shape=(1,), semantic_type="expert_id"),
        )
    )
    assert mask.shape == (4, 2)
    assert mask.semantic_type == "comparison_mask"
    assert mask.requires_grad is False


def test_forward_flops_zero() -> None:
    op = Equal()
    ctx = EstimationContext(
        port_values=(
            ("left", _resolved(Tensor(shape=(8, 4)))),
            ("right", _resolved(Tensor(shape=(1,)))),
            ("output", _resolved(Tensor(shape=(8, 4), semantic_type="comparison_mask"))),
        )
    )
    assert op.forward_flops(ctx) == 0


def test_incompatible_broadcast_raises() -> None:
    with pytest.raises(ValueError):
        Equal().infer_outputs(
            (
                Tensor(shape=(3, 4)),
                Tensor(shape=(5, 4)),
            )
        )
