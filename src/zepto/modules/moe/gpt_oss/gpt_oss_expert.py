"""Single GPT-OSS MoE expert body (clamped GLU variant)."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Maximum, Minimum, Multiply, Sigmoid

from zepto.modules.layers.affine_linear import AffineLinear


class GptOssExpert(Module):
    """One GPT-OSS expert: fused gate/up GLU with clamped activations."""

    module_kind = "GptOssExpert"

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        self.intermediate_size = intermediate_size
        self.gate_proj = AffineLinear(hidden_size, intermediate_size, bias=True)
        self.up_proj = AffineLinear(hidden_size, intermediate_size, bias=True)
        self.down_proj = AffineLinear(intermediate_size, hidden_size, bias=True)
        self._limit = Tensor(
            shape=(1,), semantic_type="constant_limit", requires_grad=False
        )
        self._neg_limit = Tensor(
            shape=(1,), semantic_type="constant_neg_limit", requires_grad=False
        )
        self._alpha = Tensor(
            shape=(1,), semantic_type="constant_alpha", requires_grad=False
        )
        self._one = Tensor(
            shape=(1,), semantic_type="constant_one", requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = Minimum()(gate, self._limit)  # type: ignore[call-arg]
        up_clamped = Minimum()(up, self._limit)  # type: ignore[call-arg]
        up = Maximum()(up_clamped, self._neg_limit)  # type: ignore[call-arg]
        glu = Multiply()(
            gate,
            Sigmoid()(Multiply()(gate, self._alpha)),  # type: ignore[call-arg]
        )
        hidden = Multiply()(Add()(up, self._one), glu)  # type: ignore[call-arg]
        return self.down_proj(hidden)  # type: ignore[return-value]


__all__ = ["GptOssExpert"]
