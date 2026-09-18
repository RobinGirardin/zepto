"""Qwen3.8 full-model compose tests."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.mixers.layer_spec import LayerSpec
from zepto.modules.mixers.mixer_config import GatedDeltaNetConfig
from zepto.modules.mixers.mixer_presets import qwen35_language_layer_specs
from zepto.modules.models.qwen38 import Qwen38, Qwen38Config
from zepto.semantic.metadata import DType


def _tiny_qwen38(seq_len: int, *, include_mtp: bool = True) -> Qwen38:
    delta = GatedDeltaNetConfig(
        hidden_size=128,
        num_qk_heads=4,
        num_v_heads=8,
        head_dim=32,
    )
    specs = (
        LayerSpec(mixer="gated_delta_net", gated_delta=delta, ffn="swiglu", swiglu_intermediate=256),
        LayerSpec(mixer="gated_delta_net", gated_delta=delta, ffn="swiglu", swiglu_intermediate=256),
    )
    mtp_tap = 1 if include_mtp else None
    return Qwen38(
        config=Qwen38Config(hidden_size=128, num_layers=2, vocab_size=512),
        seq_len=seq_len,
        include_vision=False,
        include_mtp=include_mtp,
        mtp_source_hidden_layer_index=mtp_tap,
        layer_specs=specs,
    )


def test_qwen38_delta_layer_returns_state_tuple_in_graph() -> None:
    seq_len = 8
    graph = compose_graph(
        lambda _ctx: _tiny_qwen38(seq_len, include_mtp=False),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert kinds.count("GatedDeltaNet") >= 1


def test_qwen38_primary_plus_mtp_exceeds_primary_only() -> None:
    from zepto.compose import Module

    seq_len = 8
    d = 128
    primary = compose_graph(
        lambda _ctx: _tiny_qwen38(seq_len, include_mtp=False),
        (Tensor(shape=(seq_len,)),),
    )

    class _Combined(Module):
        module_kind = "Qwen38Combined"

        def __init__(self) -> None:
            super().__init__()
            self.model = _tiny_qwen38(seq_len, include_mtp=True)

        def forward(
            self,
            token_ids: Tensor,
            mtp_ids: Tensor,
        ) -> Tensor:
            _primary, aux = self.model.forward_with_mtp(token_ids, mtp_ids)
            return aux

    combined = compose_graph(
        lambda _ctx: _Combined(),
        (
            Tensor(shape=(seq_len,)),
            Tensor(shape=(seq_len,)),
        ),
    )
    ctx = reference_invocation(default_dtype=DType.FP16)
    assert (
        estimate(combined, ctx).flops.total_flops
        > estimate(primary, ctx).flops.total_flops
    )


def test_qwen38_preset_specs_length() -> None:
    assert len(qwen35_language_layer_specs()) == 64
