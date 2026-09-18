"""Nemotron Mamba-2 mixer module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import LinearMatMul, Reshape, Split, Transpose

from .depthwise_causal_conv1d import DepthwiseCausalConv1d
from zepto.modules.layers.gated_grouped_rms_norm import GatedGroupedRMSNorm
from zepto.modules.layers.linear import Linear
from .mixer_config import Mamba2MixerConfig
from .selective_ssm_scan import SelectiveSSMScan


def _split_last(value: Tensor, sizes: tuple[int, ...]) -> tuple[Tensor, ...]:
    transposed = Transpose(permutation=(1, 0))(value)
    parts = Split(sizes=sizes)(transposed)
    return tuple(Transpose(permutation=(1, 0))(part) for part in parts)


class Mamba2Mixer(Module):
    """Mamba-2 token mixer with conv + selective scan state."""

    module_kind = "Mamba2Mixer"

    def __init__(self, config: Mamba2MixerConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config
        self.in_proj = Linear(cfg.hidden_size, cfg.in_proj_size)
        self.conv = DepthwiseCausalConv1d(
            cfg.conv_channels,
            cfg.conv_kernel_size,
            activation="silu",
            bias=cfg.conv_bias,
        )
        self.scan = SelectiveSSMScan(
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            state_size=cfg.state_size,
            num_groups=cfg.num_groups,
        )
        self.norm = GatedGroupedRMSNorm(
            cfg.intermediate_size, cfg.num_groups
        )
        self.o_proj = Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        conv_state_in: Tensor | None = None,
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError("hidden size mismatch")
        seq_len = hidden_states.shape[0]

        projected = LinearMatMul()(hidden_states, parameters=(self.in_proj.weight,))  # type: ignore[call-arg]
        z, u_xbc, delta = _split_last(
            projected,
            (
                cfg.intermediate_size,
                cfg.conv_channels,
                cfg.num_heads,
            ),
        )
        conv_out, conv_state_out = self.conv(u_xbc, conv_state_in)
        x_flat, b, c = _split_last(
            conv_out,
            (
                cfg.intermediate_size,
                cfg.num_groups * cfg.state_size,
                cfg.num_groups * cfg.state_size,
            ),
        )
        x = Reshape(shape=(seq_len, cfg.num_heads, cfg.head_dim))(x_flat)
        b_g = Reshape(shape=(seq_len, cfg.num_groups, cfg.state_size))(b)
        c_g = Reshape(shape=(seq_len, cfg.num_groups, cfg.state_size))(c)
        scan_out, scan_state_out = self.scan(x, delta, b_g, c_g, scan_state_in)
        scan_flat = Reshape(shape=(seq_len, cfg.intermediate_size))(scan_out)
        gated = self.norm(scan_flat, z)
        output = LinearMatMul()(gated, parameters=(self.o_proj.weight,))  # type: ignore[return-value]
        return output, conv_state_out, scan_state_out


__all__ = ["Mamba2Mixer"]
