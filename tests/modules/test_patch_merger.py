"""Patch merger compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.patch_merger import PatchMerger2x2


def test_qwen_merger_output_width() -> None:
    grid_h, grid_w = 8, 8
    graph = compose_graph(
        lambda _ctx: PatchMerger2x2(
            hidden_size=1152, out_dim=5120, grid_h=grid_h, grid_w=grid_w
        ),
        (Tensor(shape=(grid_h * grid_w, 1152)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == ((grid_h // 2) * (grid_w // 2), 5120)
