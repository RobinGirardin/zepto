"""Position-aware pool compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.vision.position_aware_pool import PositionAwareAveragePool2x2


def test_gemma_pool_output_length_280() -> None:
    grid_h, grid_w = 28, 40
    graph = compose_graph(
        lambda _ctx: PositionAwareAveragePool2x2(
            hidden_size=1152, out_dim=5376, grid_h=grid_h, grid_w=grid_w
        ),
        (Tensor(shape=(grid_h * grid_w, 1152)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (280, 1152)
