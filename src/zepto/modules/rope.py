"""Rotary positional-encoding module boundary."""

from zepto.compose import Module, Tensor
from zepto.semantic import Identity


class RoPE(Module):
    """Reusable rotary positional encoding composition boundary."""

    def forward(self, value: Tensor) -> Tensor:
        """Compose the current placeholder rotary-encoding behavior."""
        return Identity()(value)  # type: ignore[return-value]
