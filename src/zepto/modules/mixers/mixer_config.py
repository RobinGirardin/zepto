"""Configuration dataclasses for stateful sequence mixers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatedDeltaNetConfig:
    hidden_size: int
    num_qk_heads: int
    num_v_heads: int
    head_dim: int
    conv_kernel_size: int = 4
    conv_bias: bool = False

    @property
    def qk_proj_size(self) -> int:
        return self.num_qk_heads * self.head_dim

    @property
    def v_proj_size(self) -> int:
        return self.num_v_heads * self.head_dim

    def __post_init__(self) -> None:
        if self.num_v_heads % self.num_qk_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_qk_heads")


@dataclass(frozen=True, slots=True)
class Mamba2MixerConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    state_size: int
    num_groups: int
    conv_kernel_size: int = 4
    conv_bias: bool = True
    projection_bias: bool = False

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_channels(self) -> int:
        return self.intermediate_size + 2 * self.num_groups * self.state_size

    @property
    def in_proj_size(self) -> int:
        return (
            self.intermediate_size
            + self.conv_channels
            + self.num_heads
        )

    def __post_init__(self) -> None:
        if self.num_heads % self.num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")


__all__ = ["GatedDeltaNetConfig", "Mamba2MixerConfig"]
