"""Layer normalization module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Divide, Multiply, ReduceSum, SquareRoot, Subtract
from zepto.modules._internal._helpers import add_bias_parameter, feature_size, norm_axis, scale_by_parameter


class LayerNorm(Module):
    """Layer normalization matching HuggingFace eager semantics.

    Normalizes over the last dimension with learnable affine ``weight`` (γ) and
    ``bias`` (β) when ``elementwise_affine=True``.
    """

    module_kind = "LayerNorm"

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self._eps = Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)
        self._inv_norm_size = Tensor(
            shape=(1,), semantic_type="inv_norm_size", requires_grad=False
        )
        if elementwise_affine:
            self.weight = Parameter(shape=(normalized_shape,), semantic_type="weight")
            self.bias = Parameter(shape=(normalized_shape,), semantic_type="bias")

    def forward(self, value: Tensor) -> Tensor:
        axis = norm_axis(value)
        feature_dim = feature_size(value, axis=axis)
        if feature_dim != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected last dim {self.normalized_shape}, "
                f"got {feature_dim} from shape {value.shape}"
            )

        mean = Divide()(
            ReduceSum(axis=axis, keepdim=True)(value),
            self._inv_norm_size,
        )
        centered = Subtract()(value, mean)
        variance = Divide()(
            ReduceSum(axis=axis, keepdim=True)(Multiply()(centered, centered)),
            self._inv_norm_size,
        )
        normalized = Divide()(
            centered,
            SquareRoot()(Add()(variance, self._eps)),
        )

        if not self.elementwise_affine:
            return normalized  # type: ignore[return-value]

        return add_bias_parameter(
            scale_by_parameter(normalized, self.weight),  # type: ignore[arg-type]
            self.bias,
        )
