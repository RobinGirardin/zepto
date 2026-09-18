"""Pixel shuffle compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.vision.pixel_shuffle import PixelShuffle2x2


def test_muse_pixel_shuffle_channel_and_spatial() -> None:
    grid_h, grid_w, d = 14, 14, 1536
    graph = compose_graph(
        lambda _ctx: PixelShuffle2x2(d, grid_h=grid_h, grid_w=grid_w),
        (Tensor(shape=(grid_h * grid_w, d)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == ((grid_h // 2) * (grid_w // 2), d * 4)
