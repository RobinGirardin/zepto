"""Affine linear layer with optional bias."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import LinearMatMul, ParameterBias


class AffineLinear(Module):
    """Linear transform via activation × weight with optional bias parameter."""

    module_kind = "AffineLinear"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            shape=(in_features, out_features),
            semantic_type="weight",
        )
        if bias:
            self.bias = Parameter(shape=(out_features,), semantic_type="bias")

    def forward(self, x: Tensor) -> Tensor:
        out = LinearMatMul()(x, parameters=(self.weight,))  # type: ignore[call-arg]
        bias = getattr(self, "bias", None)
        if bias is not None:
            out = ParameterBias()(out, parameters=(bias,))  # type: ignore[call-arg,return-value]
        return out  # type: ignore[return-value]


__all__ = ["AffineLinear"]
