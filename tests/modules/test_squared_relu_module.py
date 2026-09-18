"""Compose tests for SquaredReLU module."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.squared_relu import SquaredReLU


def test_squared_relu_activation_flops() -> None:
    graph = compose_graph(
        lambda _ctx: SquaredReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    lowered = lower(graph, reference_invocation(phase="forward"), registry=registry)
    forward = sum(node.forward_flops for node in lowered.nodes)
    assert forward == 4 * 8


def test_squared_relu_graph_families() -> None:
    graph = compose_graph(
        lambda _ctx: SquaredReLU(),
        (Tensor(shape=(8, 16), requires_grad=True),),
    )
    families = tuple(graph.node(n).operation_family for n in graph.nodes)
    assert "maximum" in families
    assert "multiply" in families
