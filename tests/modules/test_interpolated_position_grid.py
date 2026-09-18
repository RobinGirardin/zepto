"""InterpolatedPositionGrid compose tests."""

from __future__ import annotations

from zepto.compose import compose_graph
from zepto.modules.position.interpolated_position_grid import InterpolatedPositionGrid


def test_interpolated_output_shape() -> None:
    graph = compose_graph(
        lambda _ctx: InterpolatedPositionGrid(
            src_h=8,
            src_w=8,
            embed_dim=64,
            out_h=4,
            out_w=4,
        ),
        (),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (16, 64)
    families = tuple(
        graph.node(node_id).operation_family for node_id in graph.nodes
    )
    assert "bilinear_resize_2d" in families
