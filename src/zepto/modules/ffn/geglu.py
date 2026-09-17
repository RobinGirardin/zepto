"""GeGLU gated feed-forward module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import GeluErfGate, GeluTanhGate, LinearMatMul, Multiply

from zepto.modules.layers.linear import Linear


class GeGLU(Module):
    """Gated FFN: ``down_proj(GELU(gate_proj(x)) * up_proj(x))``.

    Operations are inlined so every graph node shares ``GeGLU`` provenance
    for ``region/geglu`` discovery (submodule ``__call__`` would split paths).
    """

    module_kind = "GeGLU"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        gelu_approx: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if gelu_approx not in ("tanh", "none"):
            raise ValueError("gelu_approx must be 'tanh' or 'none'")

        self.gelu_approx = gelu_approx
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        gate = LinearMatMul()(x, parameters=(self.gate_proj.weight,))  # type: ignore[call-arg]
        up = LinearMatMul()(x, parameters=(self.up_proj.weight,))  # type: ignore[call-arg]
        if self.gelu_approx == "none":
            activated = GeluErfGate()(gate)
        else:
            activated = GeluTanhGate()(gate)
        hidden = Multiply()(activated, up)  # type: ignore[call-arg]
        return LinearMatMul()(hidden, parameters=(self.down_proj.weight,))  # type: ignore[return-value]
