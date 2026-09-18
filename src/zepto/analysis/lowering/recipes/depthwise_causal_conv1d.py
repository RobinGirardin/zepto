"""Closed-form cost leaf for fused depthwise causal conv1d ± SiLU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DepthwiseCausalConv1dRecipe:
    """Fused depthwise causal conv1d leaf (Dao / HF / Zepto parity)."""

    kernel_size: int = 4
    use_bias: bool = False
    activation: Literal["none", "silu"] = "silu"
    materialize_history: bool = False
    save_pre_activation: bool = True
    save_input: bool = False

    def forward_flops_per_element(self) -> int:
        base = 2 * self.kernel_size - 1 + int(self.use_bias)
        if self.activation == "silu":
            return base + 5
        return base

    def backward_flops_per_element(self) -> int:
        if self.activation == "silu":
            return 4 * self.kernel_size + 6
        return 4 * self.kernel_size - 2

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element() * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element() * numel


DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE = DepthwiseCausalConv1dRecipe(
    kernel_size=4,
    use_bias=False,
    activation="silu",
)
