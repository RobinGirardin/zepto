"""Horizon integration for GatedDeltaNet (mini config)."""

from __future__ import annotations

from zepto.analysis import (
    ConvStateConfig,
    HorizonSpec,
    RecurrentStateConfig,
    estimate_horizon,
    reference_invocation,
)
from zepto.compose import Tensor, compose_graph
from zepto.modules.gated_delta_net import GatedDeltaNet
from zepto.modules.mixer_config import GatedDeltaNetConfig
from zepto.semantic.metadata import DType


def test_gated_delta_mini_prefill_decode_horizon() -> None:
    cfg = GatedDeltaNetConfig(
        hidden_size=128,
        num_qk_heads=2,
        num_v_heads=2,
        head_dim=16,
    )
    conv_c = 2 * cfg.qk_proj_size + cfg.v_proj_size
    ctx = reference_invocation(default_dtype=DType.BF16)
    spec = HorizonSpec.inference(
        prefill=32,
        decode_steps=4,
        conv=ConvStateConfig(
            layer_indices=(0,),
            channels=conv_c,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=((0, "gated_delta", 2, 16, 16),),
            dtype=DType.BF16,
        ),
    )

    def module_fn(_compose):
        return GatedDeltaNet(cfg)

    def inputs_fn(step, _ctx, _state):
        return (Tensor(shape=(step.seq_len, cfg.hidden_size)),)

    report = estimate_horizon(spec, module_fn, inputs_fn, ctx)
    assert len(report.per_step) == 5
    assert report.state_final.custom
    assert report.per_step[1].flops.forward_flops < report.per_step[0].flops.forward_flops
