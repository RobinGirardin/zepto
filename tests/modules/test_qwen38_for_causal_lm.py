"""Qwen38ForCausalLM compose smoke tests."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor, compose_graph
from zepto.modules import Qwen38ForCausalLM
from zepto.modules.attention.attention_config import gated_gqa
from zepto.modules.mixers.layer_spec import LayerSpec
from zepto.modules.mixers.mixer_config import GatedDeltaNetConfig
from zepto.modules.models.qwen38 import Qwen38Config
from zepto.modules.position.rope_config import qwen3_vl_mrope


def _tiny_cycle_specs(*, hidden_size: int = 128, intermediate: int = 256) -> tuple[LayerSpec, ...]:
    delta = GatedDeltaNetConfig(
        hidden_size=hidden_size,
        num_qk_heads=4,
        num_v_heads=12,
        head_dim=32,
    )
    attn = gated_gqa(hidden_size, 12, 2, head_dim=64, gate="sigmoid")
    delta_layer = LayerSpec(
        mixer="gated_delta_net",
        gated_delta=delta,
        ffn="swiglu",
        swiglu_intermediate=intermediate,
    )
    attn_layer = LayerSpec(
        mixer="attention",
        attention=attn,
        ffn="swiglu",
        swiglu_intermediate=intermediate,
    )
    return (delta_layer, delta_layer, delta_layer, attn_layer)


def test_qwen38_causal_lm_compose_has_fused_ce() -> None:
    seq_len = 32
    cfg = Qwen38Config(hidden_size=128, num_layers=4, vocab_size=512)
    graph = compose_graph(
        lambda _ctx: Qwen38ForCausalLM(
            config=cfg,
            seq_len=seq_len,
            layer_specs=_tiny_cycle_specs(),
            rope_config=qwen3_vl_mrope(head_dim=64, rotary_dim=64),
        ),
        (
            Tensor(shape=(seq_len,), semantic_type="token_ids"),
            Tensor(shape=(seq_len,), semantic_type="labels"),
        ),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    gdn_paths = {
        graph.node(n).provenance.module_path
        for n in graph.nodes
        if graph.node(n).provenance.component_type == "GatedDeltaNet"
    }
    assert "FusedLinearCrossEntropy" in kinds
    assert len(gdn_paths) == 3
    assert "FlexibleAttention" in kinds or kinds.count("Qwen35LanguageDecoderBlock") >= 4
    assert kinds.count("LanguageModelOutput") == 0


def test_qwen38_causal_lm_rejects_vision_or_mtp() -> None:
    cfg = Qwen38Config(hidden_size=128, num_layers=4, vocab_size=512)
    with pytest.raises(ValueError, match="text-only"):
        Qwen38ForCausalLM(
            config=cfg,
            seq_len=32,
            layer_specs=_tiny_cycle_specs(),
            include_vision=True,
        )
    with pytest.raises(ValueError, match="text-only"):
        Qwen38ForCausalLM(
            config=cfg,
            seq_len=32,
            layer_specs=_tiny_cycle_specs(),
            include_mtp=True,
        )
