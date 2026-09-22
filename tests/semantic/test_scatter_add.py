"""Tests for ScatterAdd semantic operation."""

from __future__ import annotations

import pytest

from zepto.analysis.resolved import ResolvedValue
from zepto.compose import Module, Tensor, compose_graph
from zepto.semantic import EstimationContext, ScatterAdd
from zepto.semantic.metadata import DType


def _resolved(tensor: Tensor) -> ResolvedValue:
    return ResolvedValue(tensor=tensor, role=None, dtype=tensor.dtype or DType.FP32)


def test_infer_outputs() -> None:
    op = ScatterAdd(axis=0)
    (output,) = op.infer_outputs(
        (
            Tensor(shape=(8, 64)),
            Tensor(shape=(3,), semantic_type="routing_index"),
            Tensor(shape=(3, 64)),
        )
    )
    assert output.shape == (8, 64)


def test_duplicate_indices_allowed() -> None:
    """Duplicate indices sum contributions (torch.index_add semantics)."""
    op = ScatterAdd(axis=0)
    (output,) = op.infer_outputs(
        (
            Tensor(shape=(4, 2)),
            Tensor(shape=(2,), semantic_type="routing_index"),
            Tensor(shape=(2, 2)),
        )
    )
    assert output.shape == (4, 2)


def test_mismatch_raises() -> None:
    op = ScatterAdd(axis=0)
    with pytest.raises(ValueError, match="must match index count"):
        op.infer_outputs(
            (
                Tensor(shape=(8, 64)),
                Tensor(shape=(3,), semantic_type="routing_index"),
                Tensor(shape=(2, 64)),
            )
        )


def test_wrong_index_rank_raises() -> None:
    op = ScatterAdd(axis=0)
    with pytest.raises(ValueError, match="cannot scatter along batch axis"):
        op.infer_outputs(
            (
                Tensor(shape=(8, 64)),
                Tensor(shape=(3, 1), semantic_type="routing_index"),
                Tensor(shape=(3, 64)),
            )
        )


def test_batched_input_rank1_indices_on_token_axis() -> None:
    """MoE-style: base (B, S, d), rank-1 indices, updates (K, S, d) on axis 0."""
    op = ScatterAdd(axis=0)
    (output,) = op.infer_outputs(
        (
            Tensor(shape=(2, 8, 64)),
            Tensor(shape=(3,), semantic_type="routing_index"),
            Tensor(shape=(3, 8, 64)),
        )
    )
    assert output.shape == (2, 8, 64)


def test_batched_rank2_indices() -> None:
    op = ScatterAdd(axis=1)
    (output,) = op.infer_outputs(
        (
            Tensor(shape=(2, 8, 64)),
            Tensor(shape=(2, 3), semantic_type="routing_index"),
            Tensor(shape=(2, 3, 64)),
        )
    )
    assert output.shape == (2, 8, 64)


def test_compose_chain() -> None:
    graph = compose_graph(
        lambda _ctx: _ScatterChain(),
        (Tensor(shape=(4, 8)),),
    )
    families = tuple(graph.node(n).operation_family for n in graph.nodes)
    assert families.count("scatter_add") == 2


def test_forward_flops() -> None:
    op = ScatterAdd(axis=0)
    ctx = EstimationContext(
        port_values=(
            ("input", _resolved(Tensor(shape=(8, 64)))),
            ("indices", _resolved(Tensor(shape=(3,), semantic_type="routing_index"))),
            ("updates", _resolved(Tensor(shape=(3, 64)))),
            ("output", _resolved(Tensor(shape=(8, 64)))),
        )
    )
    assert op.forward_flops(ctx) == 3 * 64


class _ScatterChain(Module):
    module_kind = "ScatterChainFixture"

    def __init__(self) -> None:
        super().__init__()
        self._idx = Tensor(
            shape=(2,), semantic_type="routing_index", requires_grad=False
        )
        self._upd = Tensor(shape=(2, 8))
        self._upd2 = Tensor(shape=(2, 8))

    def forward(self, base: Tensor) -> Tensor:
        out = ScatterAdd(axis=0)(base, self._idx, self._upd)
        return ScatterAdd(axis=0)(out, self._idx, self._upd2)
