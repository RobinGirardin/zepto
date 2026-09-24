"""Apertus 1.5 text-only compose tests: vocab split and llama3 RoPE."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor, compose_graph
from zepto.modules.models.apertus15 import Apertus15


def _tiny_apertus15(seq_len: int) -> Apertus15:
    return Apertus15(
        hidden_size=32,
        intermediate_size=64,
        num_heads=8,
        num_kv_heads=2,
        num_layers=2,
        vocab_size=200,
        output_vocab_size=100,
        seq_len=seq_len,
        head_dim=4,
    )


def _out_shape(graph) -> tuple[int, ...]:
    return graph.edge(graph.outputs[0]).tensor.shape


def test_apertus15_embed_and_head_shapes() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = _tiny_apertus15(8)
        captured["embed"] = model.embedding.weight.shape
        captured["lm_head"] = model.lm_head.weight.shape
        return model

    compose_graph(factory, (Tensor(shape=(8,)),))
    assert captured["embed"] == (32, 200)
    assert captured["lm_head"] == (32, 100)


def test_apertus15_infer_logits_use_output_vocab() -> None:
    graph = compose_graph(
        lambda _ctx: _tiny_apertus15(8),
        (Tensor(shape=(8,)),),
    )
    assert _out_shape(graph) == (8, 100)


def test_apertus15_batched_infer_logits() -> None:
    graph = compose_graph(
        lambda _ctx: _tiny_apertus15(8),
        (Tensor(shape=(2, 8), semantic_type="token_ids", requires_grad=False),),
    )
    assert _out_shape(graph) == (2, 8, 100)


def test_apertus15_rope_is_apertus_llama3() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = _tiny_apertus15(8)
        captured["cfg"] = model.rope_materialize.config
        return model

    compose_graph(factory, (Tensor(shape=(8,)),))
    cfg = captured["cfg"]
    assert cfg.rope_type == "llama3"
    assert cfg.rope_theta == 12_000_000.0
    assert cfg.original_max_position_embeddings == 8192
    assert cfg.low_freq_factor == 1.0
    assert cfg.high_freq_factor == 4.0


def test_apertus15_8b_preset_vocab_split() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = Apertus15.apertus15_8b(seq_len=16)
        captured["embedding_vocab"] = model.embedding.vocab_size
        captured["lm_head_vocab"] = model.lm_head.vocab_size
        captured["num_layers"] = model.num_layers
        captured["seq_len"] = model.seq_len
        return model

    compose_graph(factory, (Tensor(shape=(16,)),))
    assert captured["embedding_vocab"] == 266752
    assert captured["lm_head_vocab"] == 131072
    assert captured["num_layers"] == 32
    assert captured["seq_len"] == 16


def test_apertus15_rejects_head_wider_than_embed() -> None:
    with pytest.raises(ValueError, match="output head cannot exceed input embed"):
        Apertus15(
            hidden_size=32,
            intermediate_size=64,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            vocab_size=200,
            output_vocab_size=300,
            seq_len=8,
            head_dim=4,
        )


def test_apertus15_decoder_block_count() -> None:
    graph = compose_graph(
        lambda _ctx: _tiny_apertus15(8),
        (Tensor(shape=(8,)),),
    )
    attn_paths = {
        graph.node(n).provenance.module_path
        for n in graph.nodes
        if graph.node(n).provenance.component_type == "GroupedQueryAttention"
    }
    assert attn_paths == {
        ("Apertus15", "layers.0", "attn"),
        ("Apertus15", "layers.1", "attn"),
    }
