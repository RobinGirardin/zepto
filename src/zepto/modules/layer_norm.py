"""Layer normalization module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Identity


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

    def forward(self, value: Tensor) -> Tensor:
        """Compose the current placeholder layer-normalization behavior."""
        return Identity()(value)  # type: ignore[return-value]
