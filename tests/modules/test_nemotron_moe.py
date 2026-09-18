"""Compose tests for Nemotron MoE block."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from modules.moe.moe_routing import UniformRoutingProfile
from modules.moe.nemotron.nemotron_moe_block import NemotronMoEBlock


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(n).operation_family for n in graph.nodes)


def test_nemotron_grouped_router_ops() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: NemotronMoEBlock(
            16,
            8,
            16,
            8,
            top_k=2,
            num_groups=2,
            topk_group=1,
            routing_profile=profile,
        ),
        (Tensor(shape=(2, 16), requires_grad=True),),
    )
    families = _families(graph)
    assert families.count("topk") >= 2
    assert "reshape" in families
    assert "reduce_sum" in families


def test_nemotron_relu_squared_expert_body() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: NemotronMoEBlock(
            32,
            16,
            32,
            4,
            top_k=2,
            routing_profile=profile,
        ),
        (Tensor(shape=(4, 32), requires_grad=True),),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "SquaredReLU" in kinds
    families = _families(graph)
    assert families.count("linear_matmul") >= 4
