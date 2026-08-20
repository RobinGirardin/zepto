"""Rotary positional-encoding module boundary."""

from ..core.composition import GraphTensor, Module
from ..core.functional import identity


class RoPE(Module):
    """Reusable rotary positional encoding composition boundary."""

    def forward(self, value: GraphTensor) -> GraphTensor:
        """Compose the current placeholder rotary-encoding behavior."""
        return identity(value)
