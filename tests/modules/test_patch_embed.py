"""Patch embedding compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from modules.vision.patch_embed import (
    LinearPatchEmbed,
    qwen3_vl_patch_embed,
)


def test_linear_patch_embed_output_rank() -> None:
    grid_h, grid_w, p = 2, 2, 14
    graph = compose_graph(
        lambda _ctx: LinearPatchEmbed(
            patch_size=p,
            in_channels=3,
            temporal_frames=2,
            out_dim=1536,
            grid_h=grid_h,
            grid_w=grid_w,
        ),
        (Tensor(shape=(2, grid_h * p, grid_w * p, 3)),),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (grid_h * grid_w, 1536)
    families = tuple(
        graph.node(node_id).operation_family for node_id in graph.nodes
    )
    assert "linear_matmul" in families
    assert "reshape" in families


def test_qwen_conv3d_preset_width() -> None:
    grid_t, grid_h, grid_w = 1, 4, 4
    graph = compose_graph(
        lambda _ctx: qwen3_vl_patch_embed(
            grid_t=grid_t, grid_h=grid_h, grid_w=grid_w
        ),
        (
            Tensor(
                shape=(
                    (grid_t - 1) * 2 + 2,
                    grid_h * 16,
                    grid_w * 16,
                    3,
                )
            ),
        ),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (grid_t * grid_h * grid_w, 1152)
    families = tuple(
        graph.node(node_id).operation_family for node_id in graph.nodes
    )
    assert "conv3d" in families
    assert "reshape" in families
