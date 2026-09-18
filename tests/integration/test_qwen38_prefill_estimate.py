"""Qwen3.8 prefill integration estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from modules.mixers.layer_spec import LayerSpec
from modules.mixers.mixer_config import GatedDeltaNetConfig
from modules.models.qwen38 import Qwen38, Qwen38Config
from zepto.semantic.metadata import DType


def test_qwen38_prefill_estimate_produces_cost_report() -> None:
    seq_len = 16
    delta = GatedDeltaNetConfig(
        hidden_size=128,
        num_qk_heads=4,
        num_v_heads=8,
        head_dim=32,
    )
    specs = (
        LayerSpec(
            mixer="gated_delta_net",
            gated_delta=delta,
            ffn="swiglu",
            swiglu_intermediate=256,
        ),
    )
    graph = compose_graph(
        lambda _ctx: Qwen38(
            config=Qwen38Config(hidden_size=128, num_layers=1, vocab_size=512),
            seq_len=seq_len,
            include_vision=False,
            include_mtp=False,
            layer_specs=specs,
        ),
        (Tensor(shape=(seq_len,)),),
    )
    report = estimate(
        graph,
        reference_invocation(
            requested_capabilities=frozenset({"fused", "flash"}),
            default_dtype=DType.FP16,
        ),
    )
    assert report.flops.total_flops > 0
    assert report.memory.breakdown.parameters > 0
