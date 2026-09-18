"""SiLU gate before grouped RMSNorm (Nemotron Mamba output)."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Cast, Divide, Multiply, ReduceSum, Reshape, Sigmoid, SquareRoot

from zepto.modules._internal._helpers import (
    RMSNORM_COMPUTE_DTYPE,
    activation_dtype,
    norm_axis,
    scale_by_parameter,
)


class GatedGroupedRMSNorm(Module):
    """Gate with SiLU, then independent RMSNorm over channel groups."""

    module_kind = "GatedGroupedRMSNorm"

    def __init__(
        self,
        hidden_size: int,
        num_groups: int,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_groups <= 0:
            raise ValueError("hidden_size and num_groups must be positive")
        if hidden_size % num_groups != 0:
            raise ValueError("hidden_size must be divisible by num_groups")
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_width = hidden_size // num_groups
        self.eps = eps
        self.weight = Parameter(shape=(hidden_size,), semantic_type="weight")
        self._eps = Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)
        self._inv_group_size = Tensor(
            shape=(1,), semantic_type="inv_norm_size", requires_grad=False
        )

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        if value.shape != gate.shape:
            raise ValueError("value and gate must share shape")
        axis = norm_axis(value)
        if value.shape[axis] != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got {value.shape}"
            )
        restore_dtype = activation_dtype(value)
        value_fp32 = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(value)
        gate_fp32 = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(gate)
        sig = Sigmoid()(gate_fp32)
        silu_gate = Multiply()(gate_fp32, sig)  # type: ignore[call-arg]
        gated = Multiply()(value_fp32, silu_gate)  # type: ignore[call-arg]

        seq_len = value.shape[0] if len(value.shape) == 2 else value.shape[1]
        grouped = Reshape(shape=(seq_len, self.num_groups, self.group_width))(
            gated
        )
        group_axis = 2
        squared = Multiply()(grouped, grouped)
        variance = Divide()(
            ReduceSum(axis=group_axis, keepdim=True)(squared),
            self._inv_group_size,
        )
        denom = SquareRoot()(Add()(variance, self._eps))
        normalized = Divide()(grouped, denom)
        flat = Reshape(shape=(seq_len, self.hidden_size))(normalized)
        flat = Cast(to_dtype=restore_dtype)(flat)  # type: ignore[assignment]
        return scale_by_parameter(flat, self.weight)  # type: ignore[return-value]


__all__ = ["GatedGroupedRMSNorm"]
