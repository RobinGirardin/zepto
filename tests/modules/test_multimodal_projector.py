"""Multimodal projector compose tests."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.multimodal_projector import (
    gemma4_vision_projector,
    muse_glimmer_perception_adapter,
)


def test_muse_adapter_output_width() -> None:
    seq = 32
    graph = compose_graph(
        lambda _ctx: muse_glimmer_perception_adapter(),
        (Tensor(shape=(seq, 6144)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (seq, 6656)


def test_gemma_projector_output_width() -> None:
    graph = compose_graph(
        lambda _ctx: gemma4_vision_projector(),
        (Tensor(shape=(280, 1152)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (280, 5376)


def test_projector_flops_scale_with_seq() -> None:
    small = compose_graph(
        lambda _ctx: muse_glimmer_perception_adapter(),
        (Tensor(shape=(8, 6144)),),
    )
    large = compose_graph(
        lambda _ctx: muse_glimmer_perception_adapter(),
        (Tensor(shape=(32, 6144)),),
    )
    ctx = reference_invocation(phase="forward")
    small_flops = sum(n.forward_flops for n in lower(small, ctx).nodes)
    large_flops = sum(n.forward_flops for n in lower(large, ctx).nodes)
    assert large_flops > small_flops
