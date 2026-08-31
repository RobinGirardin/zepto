"""Feed-forward network module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from .linear import Linear
from .xielu import XIELU


class FFN(Module):
    """Two-layer MLP: ``Linear(d→d_ff) → activation → Linear(d_ff→d)``.

    Default activation is :class:`XIELU` (Apertus). Pass a custom nested module
    as ``activation`` to replace it.
    """

    module_kind = "FFN"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")

        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.activation = activation if activation is not None else XIELU()

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.up_proj(value)
        activated = self.activation(hidden)
        return self.down_proj(activated)  # type: ignore[return-value]
