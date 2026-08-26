"""Root-mean-square normalization module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import add, divide, multiply, reduce_sum, scalar_input, sqrt
from ..core.metadata import ValueMetadata
from ._helpers import (
    add_bias_parameter,
    feature_size,
    norm_axis,
    require_context,
    scale_by_parameter,
)


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
        self._eps: GraphTensor | None = None
        self._inv_norm_size: GraphTensor | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        ctx = require_context()
        self._eps = scalar_input(semantic_type="epsilon")
        self._inv_norm_size = scalar_input(semantic_type="inv_norm_size")
        if self.elementwise_affine and "weight" not in self._parameters:
            self.weight = ctx.parameter(
                ValueMetadata(
                    (self.normalized_shape,),
                    semantic_type="weight",
                )
            )
        self._initialized = True

    def forward(self, value: GraphTensor) -> GraphTensor:
        self._ensure_initialized()
        assert self._eps is not None and self._inv_norm_size is not None

        axis = norm_axis(value)
        feature_dim = feature_size(value, axis=axis)
        if feature_dim != self.normalized_shape:
            raise ValueError(
                f"RMSNorm expected last dim {self.normalized_shape}, "
                f"got {feature_dim} from shape {value.metadata.shape}"
            )

        squared = multiply(value, value)
        variance = divide(
            reduce_sum(squared, axis=axis, keepdim=True),
            self._inv_norm_size,
        )
        denom = sqrt(add(variance, self._eps))
        normalized = divide(value, denom)

        if not self.elementwise_affine:
            return normalized

        return scale_by_parameter(normalized, self.weight)
