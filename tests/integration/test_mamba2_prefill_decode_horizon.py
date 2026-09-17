"""Horizon integration for Mamba2Mixer (mini config)."""

from __future__ import annotations

from zepto.analysis import (
    ConvStateConfig,
    HorizonSpec,
    RecurrentStateConfig,
    estimate_horizon,
    reference_invocation,
)
from zepto.compose import Tensor
from zepto.modules.mamba2_mixer import Mamba2Mixer
from zepto.modules.mixer_config import Mamba2MixerConfig
from zepto.semantic.metadata import DType


def test_mamba2_mini_prefill_decode_horizon() -> None:
    cfg = Mamba2MixerConfig(
        hidden_size=64,
        num_heads=2,
        head_dim=8,
        state_size=16,
        num_groups=2,
    )
    ctx = reference_invocation(default_dtype=DType.BF16)
    spec = HorizonSpec.inference(
        prefill=32,
        decode_steps=4,
        conv=ConvStateConfig(
            layer_indices=(0,),
            channels=cfg.conv_channels,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=((0, "mamba2", 2, 8, 16),),
            dtype=DType.BF16,
        ),
    )

    def module_fn(_compose):
        return Mamba2Mixer(cfg)

    def inputs_fn(step, _ctx, _state):
        return (Tensor(shape=(step.seq_len, cfg.hidden_size)),)

    report = estimate_horizon(spec, module_fn, inputs_fn, ctx)
    assert len(report.per_step) == 5
    assert report.state_final.custom
    assert report.per_step[1].flops.forward_flops < report.per_step[0].flops.forward_flops
