"""FLOP sanity: routed expert matmuls must stay far below dense T×E baseline."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.moe.moe_presets import (
    gpt_oss_moe_block,
    laguna_sparse_moe_block,
    nemotron_moe_block,
)


def _linear_matmul_forward_flops(graph) -> int:
    lowered = lower(graph, reference_invocation(phase="forward"))
    return sum(
        node.forward_flops
        for node in lowered.nodes
        if node.implementation.startswith("linear_matmul")
    )


def test_gpt_oss_flop_sanity() -> None:
    graph = compose_graph(
        lambda _ctx: gpt_oss_moe_block(seq_len=128),
        (Tensor(shape=(128, 2880), requires_grad=True),),
    )
    routed = _linear_matmul_forward_flops(graph)
    dense = 128 * 32 * 2 * 2880 * 2880
    assert routed < dense * 0.5


def test_laguna_flop_sanity() -> None:
    graph = compose_graph(
        lambda _ctx: laguna_sparse_moe_block(seq_len=32),
        (Tensor(shape=(32, 2048), requires_grad=True),),
    )
    routed = _linear_matmul_forward_flops(graph)
    dense = 32 * 256 * 2 * 2048 * 512
    assert routed < dense * 0.5


def test_nemotron_flop_sanity() -> None:
    graph = compose_graph(
        lambda _ctx: nemotron_moe_block(seq_len=64),
        (Tensor(shape=(64, 2688), requires_grad=True),),
    )
    routed = _linear_matmul_forward_flops(graph)
    dense = 64 * 128 * 2 * 2688 * 1856
    assert routed < dense * 0.5
