"""SwiGLU gated feed-forward module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import LinearMatMul, Multiply, Sigmoid

from .linear import Linear


class SwiGLU(Module):
    """Gated FFN: ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``.

    Operations are inlined so every graph node shares ``SwiGLU`` provenance
    for ``region/swiglu`` discovery (submodule ``__call__`` would split paths).
    """

    module_kind = "SwiGLU"

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")

        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        gate = LinearMatMul()(x, parameters=(self.gate_proj.weight,))  # type: ignore[call-arg]
        up = LinearMatMul()(x, parameters=(self.up_proj.weight,))  # type: ignore[call-arg]
        sig = Sigmoid()(gate)
        activated = Multiply()(gate, sig)  # type: ignore[call-arg]
        hidden = Multiply()(activated, up)  # type: ignore[call-arg]
        return LinearMatMul()(hidden, parameters=(self.down_proj.weight,))  # type: ignore[return-value]
