"""Layer normalization module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import add, divide, multiply, reduce_sum, scalar_input, sqrt, subtract
from ..core.metadata import ValueMetadata
from ._helpers import (
    add_bias_parameter,
    feature_size,
    norm_axis,
    require_context,
    scale_by_parameter,
)


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
        self._eps: GraphTensor | None = None
        self._inv_norm_size: GraphTensor | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        ctx = require_context()
        self._eps = scalar_input(semantic_type="epsilon")
        self._inv_norm_size = scalar_input(semantic_type="inv_norm_size")
        if self.elementwise_affine:
            if "weight" not in self._parameters:
                self.weight = ctx.parameter(
                    ValueMetadata((self.normalized_shape,), semantic_type="weight")
                )
            if "bias" not in self._parameters:
                self.bias = ctx.parameter(
                    ValueMetadata((self.normalized_shape,), semantic_type="bias")
                )
        self._initialized = True

    def forward(self, value: GraphTensor) -> GraphTensor:
        self._ensure_initialized()
        assert self._eps is not None and self._inv_norm_size is not None

        axis = norm_axis(value)
        feature_dim = feature_size(value, axis=axis)
        if feature_dim != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected last dim {self.normalized_shape}, "
                f"got {feature_dim} from shape {value.metadata.shape}"
            )

        mean = divide(
            reduce_sum(value, axis=axis, keepdim=True),
            self._inv_norm_size,
        )
        centered = subtract(value, mean)
        variance = divide(
            reduce_sum(multiply(centered, centered), axis=axis, keepdim=True),
            self._inv_norm_size,
        )
        normalized = divide(centered, sqrt(add(variance, self._eps)))

        if not self.elementwise_affine:
            return normalized

        return add_bias_parameter(
            scale_by_parameter(normalized, self.weight),
            self.bias,
        )
