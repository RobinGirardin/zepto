"""Root-mean-square normalization module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Divide, Multiply, ReduceSum, SquareRoot
from ._helpers import feature_size, norm_axis, scale_by_parameter


class RMSNorm(Module):
    """RMS normalization matching HuggingFace Llama eager semantics.

    Normalizes over the last dimension. With ``elementwise_affine=True`` (default)
    a learnable scale ``weight`` (γ) is applied; Apertus uses γ-only (no β).
    """

    module_kind = "RMSNorm"

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

    def forward(self, value: Tensor) -> Tensor:
        axis = norm_axis(value)
        feature_dim = feature_size(value, axis=axis)
        if feature_dim != self.normalized_shape:
            raise ValueError(
                f"RMSNorm expected last dim {self.normalized_shape}, "
                f"got {feature_dim} from shape {value.shape}"
            )

        squared = Multiply()(value, value)
        variance = Divide()(
            ReduceSum(axis=axis, keepdim=True)(squared),
            self._inv_norm_size,
        )
        denom = SquareRoot()(Add()(variance, self._eps))
        normalized = Divide()(value, denom)

        if not self.elementwise_affine:
            return normalized  # type: ignore[return-value]

        return scale_by_parameter(normalized, self.weight)  # type: ignore[arg-type]
