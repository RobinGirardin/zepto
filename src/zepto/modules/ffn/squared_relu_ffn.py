"""Two-layer MLP with Squared ReLU activation."""

from __future__ import annotations

from zepto.compose import Module, Tensor

from zepto.modules.layers.linear import Linear
from zepto.modules.layers.squared_relu import SquaredReLU


class SquaredReluFFN(Module):
    """Two-layer MLP: ``Linear(H, I) → SquaredReLU → Linear(I, H)``."""

    module_kind = "SquaredReluFFN"

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.activation = SquaredReLU()

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.up_proj(value)
        activated = self.activation(hidden)
        return self.down_proj(activated)  # type: ignore[return-value]


__all__ = ["SquaredReluFFN"]
