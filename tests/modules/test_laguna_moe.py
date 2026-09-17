"""Compose tests for Laguna sparse MoE block."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.laguna_sparse_moe_block import LagunaSparseMoEBlock
from zepto.modules.moe_routing import UniformRoutingProfile


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(n).operation_family for n in graph.nodes)


def test_laguna_moe_shared_and_routed_paths() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: LagunaSparseMoEBlock(
            2048,
            512,
            512,
            256,
            top_k=8,
            routing_profile=profile,
        ),
        (Tensor(shape=(8, 2048), requires_grad=True),),
    )
    families = _families(graph)
    assert "divide" in families or "reduce_sum" in families
    assert "add" in families
    assert families.count("linear_matmul") >= 4


def test_laguna_router_renorm_ops() -> None:
    profile = UniformRoutingProfile(tokens_per_expert=1)
    graph = compose_graph(
        lambda _ctx: LagunaSparseMoEBlock(
            2048,
            512,
            512,
            8,
            top_k=2,
            routing_profile=profile,
        ),
        (Tensor(shape=(4, 2048), requires_grad=True),),
    )
    families = _families(graph)
    assert "sigmoid" in families
    assert "topk" in families
    assert "gather" in families
