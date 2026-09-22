"""Compose tests for LogitSoftCap."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.output.logit_soft_cap import LogitSoftCap, LogitSoftCapConfig


def test_logit_soft_cap_shape_preserved() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=20.0)),
        (Tensor(shape=(8, 1024), requires_grad=True),),
    )
    out_node = graph.node(graph.node_order[-1])
    out_shape = graph.edge(out_node.output_edges[0]).tensor.shape
    assert out_shape == (8, 1024)


def test_logit_soft_cap_graph_families_and_provenance() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=30.0, output_scale=2.0)),
        (Tensor(shape=(8, 1024), requires_grad=True),),
    )
    families = {graph.node(n).operation_family for n in graph.nodes}
    assert {"divide", "tanh", "multiply"}.issubset(families)
    provenance = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert provenance == {"LogitSoftCap"}


def test_muse_and_gemma_preset_caps() -> None:
    muse = LogitSoftCapConfig(cap=20.0)
    gemma = LogitSoftCapConfig(cap=30.0)
    assert muse.cap == 20.0
    assert gemma.cap == 30.0


def test_logit_soft_cap_forward_flops_exceed_baseline() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=20.0)),
        (Tensor(shape=(8, 1024), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    capped = lower(graph, reference_invocation(phase="forward"), registry=registry)
    capped_flops = sum(node.forward_flops for node in capped.nodes)
    assert capped_flops > 8 * 1024
