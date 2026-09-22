"""Partial and full RoPEApply compose tests."""

from __future__ import annotations

import pytest

from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_config import RoPEConfig
from zepto.modules.position.rope_materialize import RoPEMaterialize

_SEQ = 8
_HEADS = 4

_APERTUS_ROPE_FAMILIES = (
    "transpose",
    "split",
    "multiply",
    "transpose",
    "transpose",
    "concat",
    "multiply",
    "multiply",
    "add",
)


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(node_id).operation_family for node_id in graph.nodes)


def _compose_rope_apply(
    head_dim: int,
    rotary_dim: int | None,
    seq: int,
    heads: int,
    *,
    batch: int | None = None,
):
    def factory(_ctx):
        mat = RoPEMaterialize(
            seq,
            head_dim,
            config=RoPEConfig(head_dim=head_dim, rotary_dim=rotary_dim),
        )
        apply = RoPEApply(head_dim, rotary_dim=rotary_dim)
        q_shape = (
            (batch, heads, seq, head_dim)
            if batch is not None
            else (heads, seq, head_dim)
        )

        class _Harness(Module):
            def __init__(self) -> None:
                super().__init__()
                self._q = Tensor(shape=q_shape, requires_grad=True)

            def forward(self) -> Tensor:
                cos, sin = mat()
                return apply(self._q, cos, sin)

        return _Harness()

    return compose_graph(factory, ())


def test_full_apply_no_split() -> None:
    graph = _compose_rope_apply(128, None, _SEQ, _HEADS)
    rope_families = _families(graph)[-len(_APERTUS_ROPE_FAMILIES) :]
    assert rope_families == _APERTUS_ROPE_FAMILIES


def test_partial_apply_split_concat() -> None:
    graph = _compose_rope_apply(256, 64, _SEQ, _HEADS)
    families = _families(graph)
    assert "split" in families
    assert "concat" in families
    out_edge = graph.edge(graph.outputs[0])
    assert out_edge.tensor.shape == (_HEADS, _SEQ, 256)


def test_wrong_cos_width_raises() -> None:
    def factory(_ctx):
        apply = RoPEApply(256, rotary_dim=64)

        class _Harness(Module):
            def __init__(self) -> None:
                super().__init__()
                self._q = Tensor(shape=(_HEADS, _SEQ, 256), requires_grad=True)
                self._cos = Tensor(shape=(_SEQ, 256), requires_grad=False)
                self._sin = Tensor(shape=(_SEQ, 64), requires_grad=False)

            def forward(self) -> Tensor:
                return apply(self._q, self._cos, self._sin)

        return _Harness()

    with pytest.raises(ValueError, match="cos/sin shape"):
        compose_graph(factory, ())


def test_batched_rank4_partial() -> None:
    graph = _compose_rope_apply(128, 64, _SEQ, _HEADS, batch=2)
    out_edge = graph.edge(graph.outputs[0])
    assert out_edge.tensor.shape == (2, _HEADS, _SEQ, 128)
