"""Last-dimension L2 normalization module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Cast, Divide, Multiply, ReduceSum, SquareRoot
from zepto.semantic.metadata import DType

from ._helpers import RMSNORM_COMPUTE_DTYPE, activation_dtype, norm_axis


class L2Normalize(Module):
    """L2-normalize each vector along the last dimension (sum-of-squares, not mean)."""

    module_kind = "L2Normalize"

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        compute_dtype: DType = RMSNORM_COMPUTE_DTYPE,
        activation_dtype_default: DType = DType.BF16,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.compute_dtype = compute_dtype
        self._activation_dtype_default = activation_dtype_default
        self._eps = Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)

    def forward(self, value: Tensor) -> Tensor:
        axis = norm_axis(value)
        restore_dtype = activation_dtype(
            value, default=self._activation_dtype_default
        )
        x = Cast(to_dtype=self.compute_dtype)(value)
        squared = Multiply()(x, x)
        norm_sq = ReduceSum(axis=axis, keepdim=True)(squared)
        denom = SquareRoot()(Add()(norm_sq, self._eps))
        normalized = Divide()(x, denom)
        return Cast(to_dtype=restore_dtype)(normalized)  # type: ignore[return-value]


__all__ = ["L2Normalize"]
