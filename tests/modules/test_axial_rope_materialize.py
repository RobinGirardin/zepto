"""AxialRoPEMaterialize graph shape tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.position.axial_rope_materialize import AxialRoPEMaterialize
from zepto.modules.position.rope_apply import RoPEApply


def test_axial_cache_shapes() -> None:
    grid_h = 4
    grid_w = 4
    head_dim = 64
    graph = compose_graph(
        lambda _ctx: AxialRoPEMaterialize(grid_h, grid_w, head_dim),
        (),
    )
    cos_shape = graph.edge(graph.outputs[0]).tensor.shape
    sin_shape = graph.edge(graph.outputs[1]).tensor.shape
    flat = grid_h * grid_w
    assert cos_shape == (flat, head_dim)
    assert sin_shape == (flat, head_dim)


def test_axial_compose_with_rope_apply() -> None:
    grid_h = 4
    grid_w = 4
    head_dim = 64
    heads = 8
    tokens = grid_h * grid_w

    def factory(_ctx):
        mat = AxialRoPEMaterialize(grid_h, grid_w, head_dim)
        apply = RoPEApply(head_dim)

        class _Harness(Module):
            def __init__(self) -> None:
                super().__init__()
                self._q = Tensor(
                    shape=(heads, tokens, head_dim),
                    requires_grad=True,
                )

            def forward(self) -> Tensor:
                cos, sin = mat()
                return apply(self._q, cos, sin)

        return _Harness()

    graph = compose_graph(factory, ())
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (heads, tokens, head_dim)
