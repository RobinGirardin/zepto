"""Nemotron H full-model compose tests."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Module, Tensor, compose_graph
from modules.attention.attention_config import AttentionConfig
from modules.mixers.layer_spec import LayerSpec
from modules.mixers.mixer_config import Mamba2MixerConfig
from modules.models.nemotron_h import NemotronH, NemotronHConfig
from zepto.semantic.metadata import DType


def _tiny_nemotron_specs() -> tuple[LayerSpec, ...]:
    mamba = Mamba2MixerConfig(
        hidden_size=128,
        num_heads=4,
        head_dim=32,
        state_size=32,
        num_groups=2,
    )
    attn = AttentionConfig(
        hidden_size=128,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        position="none",
    )
    return (
        LayerSpec(mixer="mamba2", mamba2=mamba),
        LayerSpec(mixer="moe", moe_hidden_size=128),
        LayerSpec(mixer="moe", moe_hidden_size=128),
    )


def _tiny_nemotron(seq_len: int, *, include_mtp: bool = False) -> NemotronH:
    return NemotronH(
        config=NemotronHConfig(hidden_size=128, num_layers=3, vocab_size=512),
        seq_len=seq_len,
        include_mtp=include_mtp,
        layer_specs=_tiny_nemotron_specs(),
    )


def test_nemotron_h_lowers_mamba_and_moe() -> None:
    seq_len = 8
    graph = compose_graph(
        lambda _ctx: _tiny_nemotron(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "Mamba2Mixer" in kinds
    assert "NemotronMoEBlock" in kinds


def test_nemotron_h_mtp_adds_parameters_without_primary_forward_mtp() -> None:
    seq_len = 8
    ctx = reference_invocation(default_dtype=DType.FP16)
    without = compose_graph(
        lambda _ctx: _tiny_nemotron(seq_len, include_mtp=False),
        (Tensor(shape=(seq_len,)),),
    )
    with_mtp = compose_graph(
        lambda _ctx: _tiny_nemotron(seq_len, include_mtp=True),
        (Tensor(shape=(seq_len,)),),
    )
    assert (
        estimate(with_mtp, ctx).memory.breakdown.parameters
        > estimate(without, ctx).memory.breakdown.parameters
    )
