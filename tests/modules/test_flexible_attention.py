"""Policy matrix compose tests for FlexibleAttention."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor, compose_graph
from modules.attention.attention import FlexibleAttention
from modules.attention.attention_config import (
    AttentionConfig,
    gated_gqa,
    gpt_oss_attention_layer,
    granite_attention_layer,
    llama_gqa,
)

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


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(node_id).operation_family for node_id in graph.nodes)


def _attention_core_families(graph) -> tuple[str, ...]:
    families = _families(graph)
    start = families.index("repeat_kv")
    return families[start : start + len(_ATTENTION_CORE)]


def test_llama_equiv_matches_gqa_attention_core() -> None:
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert _attention_core_families(graph) == _ATTENTION_CORE


def test_width_split_compose() -> None:
    cfg = llama_gqa(5120, 24, 4, head_dim=256)
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg),
        (
            Tensor(shape=(_SEQ, 5120), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert graph is not None
    assert cfg.q_proj_size == 6144


def test_v_equals_k_no_v_proj() -> None:
    cfg = AttentionConfig(
        hidden_size=5376,
        num_q_heads=32,
        num_kv_heads=4,
        head_dim=512,
        kv_sharing="v_equals_k",
    )
    def _factory(_ctx):
        module = FlexibleAttention(cfg)
        assert not hasattr(module, "v_proj")
        return module

    graph = compose_graph(
        _factory,
        (
            Tensor(shape=(_SEQ, 5376), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert graph is not None


def test_nope_no_rope_nodes() -> None:
    cfg = AttentionConfig(
        hidden_size=6656,
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=128,
        position="none",
    )
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg, rope=None),
        (
            Tensor(shape=(_SEQ, 6656), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert "rope" not in " ".join(_families(graph)).lower()


def test_qkv_bias_includes_parameter_bias() -> None:
    cfg = AttentionConfig(
        hidden_size=_HIDDEN,
        num_q_heads=_HEADS,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        qkv_bias=True,
        o_bias=True,
    )
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert _families(graph).count("parameter_bias") >= 4


def test_sigmoid_gate_adds_ops_before_o_proj() -> None:
    cfg = gated_gqa(6656, 32, 2, head_dim=128, gate="sigmoid")
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg),
        (
            Tensor(shape=(_SEQ, 6656), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    families = _families(graph)
    sigmoid_idx = families.index("sigmoid")
    last_matmul_idx = max(i for i, f in enumerate(families) if f == "matmul")
    assert sigmoid_idx > last_matmul_idx
    assert "multiply" in families[sigmoid_idx:]


def test_sink_softmax_uses_dedicated_family() -> None:
    cfg = gpt_oss_attention_layer(layer_index=0)
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg),
        (
            Tensor(shape=(_SEQ, 2880), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    families = _families(graph)
    assert "attention_softmax_with_sink" in families
    assert "exp" not in families


def test_batched_rank3_hidden_states() -> None:
    cfg = granite_attention_layer()
    batch = 2
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(cfg),
        (
            Tensor(shape=(batch, _SEQ, 4096), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    assert graph is not None


def test_invalid_rank_raises() -> None:
    cfg = llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
    with pytest.raises(ValueError, match="rank"):
        compose_graph(
            lambda _ctx: FlexibleAttention(cfg),
            (
                Tensor(shape=(_SEQ,), requires_grad=True),
                Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
            ),
        )
