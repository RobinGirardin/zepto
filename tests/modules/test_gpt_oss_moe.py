"""Compose tests for GPT-OSS MoE block."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.gpt_oss_moe_block import GptOssMoEBlock
from zepto.modules.moe_routing import UniformRoutingProfile


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(n).operation_family for n in graph.nodes)


def test_gpt_oss_moe_compose() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: GptOssMoEBlock(
            2880,
            2880,
            32,
            top_k=4,
            routing_profile=profile,
        ),
        (Tensor(shape=(8, 2880), requires_grad=True),),
    )
    families = _families(graph)
    assert "topk" in families
    assert "gather" in families
    assert "scatter_add" in families
    assert "divide" in families or "reduce_sum" in families


def test_gpt_oss_moe_routed_flops_below_dense() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: GptOssMoEBlock(
            2880,
            2880,
            32,
            top_k=4,
            routing_profile=profile,
        ),
        (Tensor(shape=(8, 2880), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation(phase="forward"))
    routed = sum(
        node.forward_flops
        for node in lowered.nodes
        if node.implementation.startswith("linear_matmul")
    )
    t, h, e, i, k = 8, 2880, 32, 2880, 4
    dense = t * e * 2 * h * i
    assert routed < 0.5 * dense
