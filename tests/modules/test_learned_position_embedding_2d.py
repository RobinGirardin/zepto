"""LearnedPositionEmbedding2D compose tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from modules.position.learned_position_embedding_2d import LearnedPositionEmbedding2D
from zepto.semantic import Add


def test_learned_position_output_shape() -> None:
    grid_h = 4
    grid_w = 4
    embed_dim = 128
    graph = compose_graph(
        lambda _ctx: LearnedPositionEmbedding2D(
            max_grid_h=16,
            max_grid_w=16,
            embed_dim=embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        ),
        (),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (grid_h * grid_w, embed_dim)


def test_compose_with_add() -> None:
    grid_h = 2
    grid_w = 2
    embed_dim = 8

    def factory(_ctx):
        mod = LearnedPositionEmbedding2D(
            max_grid_h=8,
            max_grid_w=8,
            embed_dim=embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        class _Harness(Module):
            def __init__(self) -> None:
                super().__init__()
                self._patches = Tensor(
                    shape=(grid_h * grid_w, embed_dim),
                    requires_grad=True,
                )

            def forward(self) -> Tensor:
                return Add()(self._patches, mod())

        return _Harness()

    graph = compose_graph(factory, ())
    assert graph.edge(graph.outputs[0]).tensor.shape == (4, embed_dim)
