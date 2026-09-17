"""Backward compatibility: GroupedQueryAttention ≡ FlexibleAttention(llama_gqa)."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.attention import FlexibleAttention
from zepto.modules.attention_config import llama_gqa
from zepto.modules.gqa import GroupedQueryAttention

_SEQ = 8
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_HIDDEN = _HEADS * _HEAD_DIM

_ATTENTION_CORE = (
    "repeat_kv",
    "repeat_kv",
    "transpose",
    "matmul",
    "add",
    "multiply",
    "exp",
    "reduce_sum",
    "divide",
    "matmul",
)


def _attention_core_families(graph) -> tuple[str, ...]:
    families = tuple(graph.node(n).operation_family for n in graph.nodes)
    start = families.index("repeat_kv")
    return families[start : start + len(_ATTENTION_CORE)]


def test_gqa_is_flexible_attention_subclass() -> None:
    assert issubclass(GroupedQueryAttention, FlexibleAttention)
    graph = compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert graph is not None


def test_gqa_graph_matches_flexible_llama_preset() -> None:
    inputs = (
        Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
        Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
    )
    gqa_graph = compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM
        ),
        inputs,
    )
    flex_graph = compose_graph(
        lambda _ctx: FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        ),
        inputs,
    )
    assert _attention_core_families(gqa_graph) == _ATTENTION_CORE
    assert _attention_core_families(flex_graph) == _ATTENTION_CORE
    assert _attention_core_families(gqa_graph) == _attention_core_families(
        flex_graph
    )
