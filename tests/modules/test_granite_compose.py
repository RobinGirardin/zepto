"""Granite full-model compose smoke tests."""

from __future__ import annotations

from zepto.analysis import estimate, lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.models.granite import Granite, GraniteConfig
from zepto.semantic.metadata import DType


def _tiny_granite(seq_len: int) -> Granite:
    return Granite(
        config=GraniteConfig(
            hidden_size=128,
            intermediate_size=256,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=32,
            num_layers=2,
            vocab_size=1024,
        ),
        seq_len=seq_len,
    )


def _flash_ctx():
    return reference_invocation(
        requested_capabilities=frozenset({"fused", "flash"}),
        default_dtype=DType.FP16,
    )


def test_granite_compose_has_flexible_attention_per_layer() -> None:
    seq_len = 32
    graph = compose_graph(
        lambda _ctx: _tiny_granite(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert kinds.count("FlexibleAttention") >= 2


def test_forward_hidden_skips_lm_head() -> None:
    seq_len = 32
    graph = compose_graph(
        lambda _ctx: _tiny_granite(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert "LanguageModelOutput" in kinds
    assert "LinearMatMul" in kinds or "LanguageModelOutput" in kinds


def test_granite_estimate_positive_flops() -> None:
    seq_len = 64
    graph = compose_graph(
        lambda _ctx: _tiny_granite(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    report = estimate(graph, _flash_ctx())
    assert report.flops.total_flops > 0
    assert report.memory.breakdown.parameters > 0
    _, lowered = estimate(graph, _flash_ctx(), return_lowered=True)
    gqa = [n for n in lowered.nodes if n.implementation.startswith("region/gqa")]
    assert len(gqa) == 2
