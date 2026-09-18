"""Closed-form cost envelope for fused GatedDeltaNet mixer block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from modules.mixers.mixer_config import GatedDeltaNetConfig

from .depthwise_causal_conv1d import DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE
from .gated_delta_scan import DEFAULT_GATED_DELTA_SCAN_RECIPE
from .gated_rms_norm import DEFAULT_GATED_RMS_NORM_RECIPE
from .l2_normalize import DEFAULT_L2_NORMALIZE_RECIPE


@dataclass(frozen=True, slots=True)
class GatedDeltaNetRecipe:
    """Block envelope = Tier-A fused leaf sum (research §4.2 / §5)."""

    conv_kernel_size: int = 4
    conv_bias: bool = False
    conv_activation: Literal["silu"] = "silu"
    use_qk_l2norm_in_kernel: bool = False
    use_gate_in_kernel: bool = False
    save_l2_rstd: bool = True
    save_conv_pre_activation: bool = True
    save_scan_state_checkpoint: bool = True
    save_gated_rms_rstd: bool = True

    def _geometry(
        self, cfg: GatedDeltaNetConfig, seq_len: int
    ) -> dict[str, int]:
        del seq_len
        d = cfg.hidden_size
        h_k = cfg.num_qk_heads
        h_v = cfg.num_v_heads
        d_k = cfg.head_dim
        d_v = cfg.head_dim
        d_q = cfg.qk_proj_size
        d_v_proj = cfg.v_proj_size
        channels = 2 * d_q + d_v_proj
        return {
            "d": d,
            "h_k": h_k,
            "h_v": h_v,
            "d_k": d_k,
            "d_v": d_v,
            "d_Q": d_q,
            "d_V": d_v_proj,
            "C": channels,
        }

    def _glue_forward(self, *, seq_len: int, h_v: int, d_k: int) -> int:
        return seq_len * h_v * (d_k + 11)

    def _glue_backward(self, *, seq_len: int, h_v: int, d_k: int) -> int:
        return 2 * self._glue_forward(seq_len=seq_len, h_v=h_v, d_k=d_k)

    def forward_flops(
        self, cfg: GatedDeltaNetConfig, *, seq_len: int
    ) -> int:
        g = self._geometry(cfg, seq_len)
        s, d = seq_len, g["d"]
        h_k, h_v = g["h_k"], g["h_v"]
        d_k, d_v = g["d_k"], g["d_v"]
        d_q, d_v_proj, channels = g["d_Q"], g["d_V"], g["C"]

        linear = (
            4 * s * d * (d_q + d_v_proj)
            + 4 * s * d * h_v
            + 2 * s * d_v_proj * d
        )
        conv = DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE.forward_flops(s * channels)
        l2_pair = 2 * DEFAULT_L2_NORMALIZE_RECIPE.forward_flops(s * h_k * d_k)
        glue = self._glue_forward(seq_len=s, h_v=h_v, d_k=d_k)
        scan = DEFAULT_GATED_DELTA_SCAN_RECIPE.forward_flops(
            seq_len=s,
            num_heads=h_v,
            key_dim=d_k,
            value_dim=d_v,
        )
        gated_rms = DEFAULT_GATED_RMS_NORM_RECIPE.forward_flops(s * h_v * d_v)
        total = linear + conv + l2_pair + glue + scan + gated_rms
        if self.use_qk_l2norm_in_kernel:
            total -= 6 * s * h_k * d_k
        if self.use_gate_in_kernel:
            total -= 10 * s * h_v * d_v
        return total

    def backward_flops(
        self,
        cfg: GatedDeltaNetConfig,
        *,
        seq_len: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        g = self._geometry(cfg, seq_len)
        s, d = seq_len, g["d"]
        h_k, h_v = g["h_k"], g["h_v"]
        d_k, d_v = g["d_k"], g["d_v"]
        d_q, d_v_proj, channels = g["d_Q"], g["d_V"], g["C"]

        linear = 2 * (
            4 * s * d * (d_q + d_v_proj)
            + 4 * s * d * h_v
            + 2 * s * d_v_proj * d
        )
        conv = DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE.backward_flops(
            s * channels, requires_grad=True
        )
        l2_pair = 2 * DEFAULT_L2_NORMALIZE_RECIPE.backward_flops(
            s * h_k * d_k, requires_grad=True
        )
        glue = self._glue_backward(seq_len=s, h_v=h_v, d_k=d_k)
        scan = DEFAULT_GATED_DELTA_SCAN_RECIPE.backward_flops(
            seq_len=s,
            num_heads=h_v,
            key_dim=d_k,
            value_dim=d_v,
            requires_grad=True,
        )
        gated_rms = DEFAULT_GATED_RMS_NORM_RECIPE.backward_flops(
            s * h_v * d_v, requires_grad=True
        )
        total = linear + conv + l2_pair + glue + scan + gated_rms
        if self.use_qk_l2norm_in_kernel:
            total -= 8 * s * h_k * d_k
        if self.use_gate_in_kernel:
            total -= 14 * s * h_v * d_v
        return total

    def decode_forward_flops(self, cfg: GatedDeltaNetConfig) -> int:
        return self.forward_flops(cfg, seq_len=1)

    def decode_backward_flops(
        self, cfg: GatedDeltaNetConfig, *, requires_grad: bool
    ) -> int:
        return self.backward_flops(cfg, seq_len=1, requires_grad=requires_grad)


DEFAULT_GATED_DELTA_NET_RECIPE = GatedDeltaNetRecipe()

FLA_LAYER_PARITY_GATED_DELTA_NET_RECIPE = GatedDeltaNetRecipe(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=False,
    save_l2_rstd=False,
)

DECODE_GATED_DELTA_NET_RECIPE = DEFAULT_GATED_DELTA_NET_RECIPE
