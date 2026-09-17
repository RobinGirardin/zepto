"""Qwen3-VL vision tower prefill cost estimation."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.vision_presets import Qwen3VLVisionTower


def _tower(grid_t: int, grid_h: int, grid_w: int) -> Qwen3VLVisionTower:
    return Qwen3VLVisionTower(
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        num_layers=2,
    )


def _pixels(grid_t: int, grid_h: int, grid_w: int) -> Tensor:
    t_in = (grid_t - 1) * 2 + 2
    return Tensor(shape=(t_in, grid_h * 16, grid_w * 16, 3))


def test_qwen_vision_flops_increase_with_grid() -> None:
    small = compose_graph(
        lambda _ctx: _tower(1, 4, 4),
        (_pixels(1, 4, 4),),
    )
    large = compose_graph(
        lambda _ctx: _tower(1, 8, 8),
        (_pixels(1, 8, 8),),
    )
    ctx = reference_invocation()
    small_report = estimate(small, ctx)
    large_report = estimate(large, ctx)
    assert large_report.flops.total_flops > small_report.flops.total_flops
