"""GraniteForCausalLM compose smoke tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules import GraniteForCausalLM
from zepto.modules.models.granite import GraniteConfig


def test_granite_causal_lm_compose_has_fused_ce() -> None:
    seq_len = 32
    cfg = GraniteConfig(
        hidden_size=128, intermediate_size=256,
        num_q_heads=4, num_kv_heads=2, head_dim=32,
        num_layers=2, vocab_size=1024,
    )
    graph = compose_graph(
        lambda _ctx: GraniteForCausalLM(config=cfg, seq_len=seq_len),
        (
            Tensor(shape=(seq_len,), semantic_type="token_ids"),
            Tensor(shape=(seq_len,), semantic_type="labels"),
        ),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert "FusedLinearCrossEntropy" in kinds
    assert "SwiGLUDecoderBlock" in kinds or kinds.count("SwiGLU") >= 1
    # Train forward uses FLCE with the shared weight; lm_output is not invoked.
    assert kinds.count("LanguageModelOutput") == 0
