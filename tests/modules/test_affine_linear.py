"""Tests for AffineLinear module composition."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.affine_linear import AffineLinear


def _families(graph) -> list[str]:
    return [graph.node(node_id).operation_family for node_id in graph.nodes]


def test_affine_linear_without_bias() -> None:
    graph = compose_graph(
        lambda _ctx: AffineLinear(64, 128),
        (Tensor(shape=(16, 64), requires_grad=True),),
    )
    assert "linear_matmul" in _families(graph)
    assert "parameter_bias" not in _families(graph)


def test_affine_linear_with_bias() -> None:
    graph = compose_graph(
        lambda _ctx: AffineLinear(64, 128, bias=True),
        (Tensor(shape=(16, 64), requires_grad=True),),
    )
    families = _families(graph)
    assert families.count("linear_matmul") == 1
    assert families.count("parameter_bias") == 1
