"""Gemma 4 vision path prefill cost estimation."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from modules.vision.vision_presets import Gemma4VisionPath


def test_gemma4_vision_pool_and_projector() -> None:
    grid_h, grid_w = 28, 40
    graph = compose_graph(
        lambda _ctx: Gemma4VisionPath(
            grid_h=grid_h, grid_w=grid_w, num_layers=2
        ),
        (Tensor(shape=(1, grid_h * 16, grid_w * 16, 3)),),
    )
    report = estimate(graph, reference_invocation())
    assert report.flops.total_flops > 0
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (280, 5376)
