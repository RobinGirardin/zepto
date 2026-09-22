"""Compose tests for GptOssExpert."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.moe.gpt_oss.gpt_oss_expert import GptOssExpert


def test_gpt_oss_expert_op_sequence() -> None:
    graph = compose_graph(
        lambda _ctx: GptOssExpert(32, 16),
        (Tensor(shape=(4, 32), requires_grad=True),),
    )
    families = tuple(graph.node(n).operation_family for n in graph.nodes)
    assert families.count("linear_matmul") == 3
    assert "minimum" in families
    assert "maximum" in families
    assert "sigmoid" in families


def test_clamp_ops_do_not_inflate_flops() -> None:
    graph = compose_graph(
        lambda _ctx: GptOssExpert(16, 8),
        (Tensor(shape=(2, 16), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation(phase="forward"))
    clamp_flops = sum(
        node.forward_flops
        for node in lowered.nodes
        if node.implementation.split("/")[0] in {"minimum", "maximum"}
    )
    assert clamp_flops == 0
