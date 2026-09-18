"""Single SwiGLU hidden layer for Qwen-style MTP heads."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import LinearMatMul, Multiply, Sigmoid

from zepto.modules.layers.linear import Linear


class MtpMlpBlock(Module):
    """One SwiGLU FFN: ``u → SwiGLU(u)`` with no residual."""

    module_kind = "MtpMlpBlock"

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, u: Tensor) -> Tensor:
        if len(u.shape) != 2:
            raise ValueError(f"MtpMlpBlock expects rank-2 input (S, d), got {u.shape}")
        if u.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden_size}, got {u.shape[1]}"
            )
        gate = LinearMatMul()(u, parameters=(self.gate_proj.weight,))  # type: ignore[call-arg]
        up = LinearMatMul()(u, parameters=(self.up_proj.weight,))  # type: ignore[call-arg]
        sig = Sigmoid()(gate)
        activated = Multiply()(gate, sig)  # type: ignore[call-arg]
        hidden = Multiply()(activated, up)  # type: ignore[call-arg]
        return LinearMatMul()(hidden, parameters=(self.down_proj.weight,))  # type: ignore[return-value]


__all__ = ["MtpMlpBlock"]
