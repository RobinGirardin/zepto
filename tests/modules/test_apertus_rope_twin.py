"""Zepto Apertus uses checkpoint-faithful llama3 RoPE."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.models.apertus import Apertus


def test_apertus_rope_materialize_uses_llama3() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = Apertus(
            hidden_size=32,
            intermediate_size=64,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            vocab_size=100,
            seq_len=8,
            head_dim=4,
        )
        captured["cfg"] = model.rope_materialize.config
        return model

    compose_graph(factory, (Tensor(shape=(8,)),))
    cfg = captured["cfg"]
    assert cfg.rope_type == "llama3"
    assert cfg.rope_theta == 12_000_000.0
    assert cfg.original_max_position_embeddings == 8192
    assert cfg.low_freq_factor == 1.0
    assert cfg.high_freq_factor == 4.0
