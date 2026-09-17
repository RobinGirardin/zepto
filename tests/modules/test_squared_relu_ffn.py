"""Compose tests for SquaredReluFFN module."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from modules.ffn.squared_relu_ffn import SquaredReluFFN


def test_squared_relu_ffn_output_shape() -> None:
    graph = compose_graph(
        lambda _ctx: SquaredReluFFN(64, 32),
        (Tensor(shape=(8, 64), requires_grad=True),),
    )
    out_tensor = graph.edge(graph.outputs[0]).tensor
    assert out_tensor.shape == (8, 64)


def test_squared_relu_ffn_families() -> None:
    graph = compose_graph(
        lambda _ctx: SquaredReluFFN(16, 8),
        (Tensor(shape=(4, 16), requires_grad=True),),
    )
    families = tuple(graph.node(n).operation_family for n in graph.nodes)
    assert families.count("linear_matmul") == 2
    assert "maximum" in families
    assert "multiply" in families
