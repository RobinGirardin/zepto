"""Muse Glimmer vision tower prefill cost estimation."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.vision_presets import MuseGlimmerVisionTower


def test_muse_vision_estimate_families() -> None:
    grid = 14
    graph = compose_graph(
        lambda _ctx: MuseGlimmerVisionTower(grid_h=grid, grid_w=grid, num_layers=2),
        (Tensor(shape=(2, grid * 14, grid * 14, 3)),),
    )
    report = estimate(graph, reference_invocation())
    assert report.flops.total_flops > 0
    families = {
        graph.node(node_id).operation_family for node_id in graph.nodes
    }
    assert "linear_matmul" in families
    assert "reshape" in families
