"""End-to-end RoPE preset compose tests with FlexibleAttention."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.attention import FlexibleAttention
from zepto.modules.attention_config import gated_gqa
from zepto.modules.multimodal_rope_materialize import MultimodalRoPEMaterialize
from zepto.modules.rope_apply import RoPEApply
from zepto.modules.rope_config import (
    LayerRoPEBinding,
    gemma4_layer_binding,
    gpt_oss_layer_binding,
    laguna_layer_binding,
    muse_glimmer_layer_binding,
    qwen3_vl_mrope,
)
from zepto.modules.rope_materialize import RoPEMaterialize

_SEQ = 8


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(node_id).operation_family for node_id in graph.nodes)


def _compose_attention_with_rope(binding: LayerRoPEBinding):
    attn_cfg = binding.attention
    hidden = attn_cfg.hidden_size

    def factory(_ctx):
        rope_mod = None
        if binding.rope is not None:
            rope_mod = RoPEApply(
                attn_cfg.head_dim,
                rotary_dim=binding.rope.resolved_rotary_dim,
            )
        attn = FlexibleAttention(attn_cfg, rope=rope_mod)
        mat = (
            RoPEMaterialize(_SEQ, attn_cfg.head_dim, config=binding.rope)
            if binding.rope
            else None
        )

        class _Harness(Module):
            def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
                if mat is None:
                    return attn(hidden_states, mask)
                cos, sin = mat()
                return attn(hidden_states, mask, cos, sin)

        return _Harness()

    return compose_graph(
        factory,
        (
            Tensor(shape=(_SEQ, hidden), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def test_gpt_oss_layer_binding() -> None:
    graph = _compose_attention_with_rope(gpt_oss_layer_binding(layer_index=0))
    assert "attention_softmax_with_sink" in _families(graph)


def _cos_cache_width(graph) -> int:
    for edge_id, edge in graph.edges.items():
        if edge.tensor.shape and len(edge.tensor.shape) == 2:
            producer = edge.producer
            if producer is not None:
                node = graph.node(producer.node_id)
                if node.operation_family == "cos":
                    return edge.tensor.shape[1]
    raise AssertionError("cos cache edge not found")


def test_laguna_layer_binding_partial_and_full() -> None:
    graph0 = _compose_attention_with_rope(laguna_layer_binding(layer_index=0))
    assert _cos_cache_width(graph0) == 64
    graph1 = _compose_attention_with_rope(laguna_layer_binding(layer_index=1))
    assert _cos_cache_width(graph1) == 128


def test_gemma4_layer_binding() -> None:
    _compose_attention_with_rope(gemma4_layer_binding(layer_index=0))
    graph5 = _compose_attention_with_rope(gemma4_layer_binding(layer_index=5))
    assert "split" in _families(graph5)


def test_muse_glimmer_nope_layer() -> None:
    graph3 = _compose_attention_with_rope(muse_glimmer_layer_binding(layer_index=3))
    assert "cos" not in _families(graph3)
    graph0 = _compose_attention_with_rope(muse_glimmer_layer_binding(layer_index=0))
    assert "cos" in _families(graph0)


def test_qwen_mrope_with_flexible_attention() -> None:
    cfg = gated_gqa(5120, 24, 4, head_dim=256)
    rope_cfg = qwen3_vl_mrope()

    def factory(_ctx):
        mat = MultimodalRoPEMaterialize(_SEQ, rope_cfg)
        rope = RoPEApply(256, rotary_dim=64)
        attn = FlexibleAttention(cfg, rope=rope)

        class _Harness(Module):
            def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
                cos, sin = mat()
                return attn(hidden_states, mask, cos, sin)

        return _Harness()

    graph = compose_graph(
        factory,
        (
            Tensor(shape=(_SEQ, 5120), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    cos_outputs = [
        graph.edge(edge_id).tensor.shape
        for edge_id in graph.outputs
    ]
    assert (_SEQ, 64) in cos_outputs or any(
        graph.edge(edge_id).tensor.shape == (_SEQ, 64)
        for edge_id in graph.edges
    )
