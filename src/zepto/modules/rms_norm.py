"""Root-mean-square normalization module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Cast, Divide, Multiply, ReduceSum, SquareRoot, Subtract
from zepto.semantic.metadata import DType
from ._helpers import (
    RMSNORM_COMPUTE_DTYPE,
    activation_dtype,
    feature_size,
    norm_axis,
    scale_by_parameter,
)


class RMSNorm(Module):
    """RMS normalization matching HuggingFace Llama eager semantics.

    Normalizes over the last dimension. Variance accumulation runs in fp32
    (``Cast`` up/down), matching ``LlamaRMSNorm`` / ``ApertusRMSNorm`` eager.
    With ``elementwise_affine=True`` (default) a learnable scale ``weight``
    (γ) is applied; Apertus uses γ-only (no β). With ``center=True``, the mean
    along the normalized axis is subtracted before RMS (Muse/Gemma centered norm).
    """

    module_kind = "RMSNorm"

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        center: bool = False,
        compute_dtype: DType = RMSNORM_COMPUTE_DTYPE,
        activation_dtype_default: DType = DType.BF16,
    ) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.center = center
        self.compute_dtype = compute_dtype
        self._activation_dtype_default = activation_dtype_default

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

        restore_dtype = activation_dtype(
            value, default=self._activation_dtype_default
        )

        # HF eager: hidden_states = hidden_states.to(float32)
        x_compute = Cast(to_dtype=self.compute_dtype)(value)

        if self.center:
            mean = Divide()(
                ReduceSum(axis=axis, keepdim=True)(x_compute),
                self._inv_norm_size,
            )
            x_compute = Subtract()(x_compute, mean)  # type: ignore[assignment]

        squared = Multiply()(x_compute, x_compute)
        variance = Divide()(
            ReduceSum(axis=axis, keepdim=True)(squared),
            self._inv_norm_size,
        )
        denom = SquareRoot()(Add()(variance, self._eps))

        # Normalize in fp32 using x_compute (not raw value)
        normalized = Divide()(x_compute, denom)

        # HF eager: hidden_states.to(input_dtype) before × γ
        normalized = Cast(to_dtype=restore_dtype)(normalized)  # type: ignore[assignment]

        if not self.elementwise_affine:
            return normalized  # type: ignore[return-value]

        return scale_by_parameter(normalized, self.weight)  # type: ignore[arg-type]
