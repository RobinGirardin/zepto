"""Gated RMSNorm: normalize, scale, then SiLU gate (Qwen DeltaNet output)."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Cast, Divide, Multiply, ReduceSum, Sigmoid, SquareRoot

from zepto.modules._internal._helpers import (
    RMSNORM_COMPUTE_DTYPE,
    activation_dtype,
    feature_size,
    norm_axis,
    scale_by_parameter,
)


class GatedRMSNorm(Module):
    """Per-head RMSNorm followed by SiLU gate multiplication."""

    module_kind = "GatedRMSNorm"

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-6,
        compute_dtype=RMSNORM_COMPUTE_DTYPE,
    ) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.compute_dtype = compute_dtype
        self.weight = Parameter(shape=(normalized_shape,), semantic_type="weight")
        self._eps = Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)
        self._inv_norm_size = Tensor(
            shape=(1,), semantic_type="inv_norm_size", requires_grad=False
        )

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        if value.shape != gate.shape:
            raise ValueError("GatedRMSNorm value and gate must share shape")
        axis = norm_axis(value)
        if feature_size(value, axis=axis) != self.normalized_shape:
            raise ValueError(
                f"expected last dim {self.normalized_shape}, got {value.shape}"
            )
        restore_dtype = activation_dtype(value)
        value_fp32 = Cast(to_dtype=self.compute_dtype)(value)
        gate_fp32 = Cast(to_dtype=self.compute_dtype)(gate)

        squared = Multiply()(value_fp32, value_fp32)
        variance = Divide()(
            ReduceSum(axis=axis, keepdim=True)(squared),
            self._inv_norm_size,
        )
        denom = SquareRoot()(Add()(variance, self._eps))
        normalized = Divide()(value_fp32, denom)
        normalized = Cast(to_dtype=restore_dtype)(normalized)  # type: ignore[assignment]
        scaled = scale_by_parameter(normalized, self.weight)  # type: ignore[arg-type]

        sig = Sigmoid()(gate_fp32)
        silu_gate = Multiply()(gate_fp32, sig)  # type: ignore[call-arg]
        return Multiply()(scaled, silu_gate)  # type: ignore[return-value]


__all__ = ["GatedRMSNorm"]
