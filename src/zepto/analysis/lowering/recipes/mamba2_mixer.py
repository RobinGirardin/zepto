"""Closed-form cost envelope for fused Mamba-2 mixer block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modules.mixers.mixer_config import Mamba2MixerConfig

from .depthwise_causal_conv1d import DepthwiseCausalConv1dRecipe
from .gated_grouped_rms_norm import DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE
from .mamba2_scan import DEFAULT_MAMBA2_SCAN_RECIPE


@dataclass(frozen=True, slots=True)
class Mamba2MixerRecipe:
    """Block envelope = Tier-A fused leaf sum (research §4.1 / §5.2)."""

    conv_kernel_size: int = 4
    conv_bias: bool = True
    conv_activation: Literal["silu"] = "silu"
    save_conv_pre_activation: bool = True
    save_scan_state_checkpoint: bool = True
    save_group_rstd: bool = True
    hub_mega_fused_execution: bool = False

    def _conv_recipe(self) -> DepthwiseCausalConv1dRecipe:
        return DepthwiseCausalConv1dRecipe(
            kernel_size=self.conv_kernel_size,
            use_bias=self.conv_bias,
            activation=self.conv_activation,
        )

    def forward_flops(self, cfg: Mamba2MixerConfig, *, seq_len: int) -> int:
        s = seq_len
        d, h, p, n, g = (
            cfg.hidden_size,
            cfg.num_heads,
            cfg.head_dim,
            cfg.state_size,
            cfg.num_groups,
        )
        del g
        m = cfg.intermediate_size
        c_c = cfg.conv_channels
        p_in = cfg.in_proj_size
        linears = 2 * s * d * p_in + 2 * s * m * d
        conv = self._conv_recipe().forward_flops(s * c_c)
        scan = DEFAULT_MAMBA2_SCAN_RECIPE.forward_flops(
            seq_len=s, num_heads=h, head_dim=p, state_size=n
        )
        norm = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE.forward_flops(s * m)
        return linears + conv + scan + norm

    def backward_flops(
        self,
        cfg: Mamba2MixerConfig,
        *,
        seq_len: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        s = seq_len
        d, h, p, n = (
            cfg.hidden_size,
            cfg.num_heads,
            cfg.head_dim,
            cfg.state_size,
        )
        m = cfg.intermediate_size
        c_c = cfg.conv_channels
        p_in = cfg.in_proj_size
        linears = 4 * s * d * p_in + 4 * s * m * d
        conv = self._conv_recipe().backward_flops(s * c_c, requires_grad=True)
        scan = DEFAULT_MAMBA2_SCAN_RECIPE.backward_flops(
            seq_len=s,
            num_heads=h,
            head_dim=p,
            state_size=n,
            requires_grad=True,
        )
        norm = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE.backward_flops(
            s * m, requires_grad=True
        )
        return linears + conv + scan + norm

    def decode_forward_flops(self, cfg: Mamba2MixerConfig) -> int:
        return self.forward_flops(cfg, seq_len=1)

    def decode_backward_flops(
        self, cfg: Mamba2MixerConfig, *, requires_grad: bool
    ) -> int:
        return self.backward_flops(cfg, seq_len=1, requires_grad=requires_grad)


DEFAULT_MAMBA2_MIXER_RECIPE = Mamba2MixerRecipe()

HUB_MEGA_MAMBA2_MIXER_RECIPE = Mamba2MixerRecipe(hub_mega_fused_execution=True)

DECODE_MAMBA2_MIXER_RECIPE = DEFAULT_MAMBA2_MIXER_RECIPE
