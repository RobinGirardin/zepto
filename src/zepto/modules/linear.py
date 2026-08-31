"""Linear layer module."""

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import LinearMatMul


class Linear(Module):
    """Affine transform via activation × weight parameter (weight-only)."""

    module_kind = "Linear"

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(
            shape=(in_features, out_features),
            semantic_type="weight",
        )

    def forward(self, x: Tensor) -> Tensor:
        return LinearMatMul()(x, parameters=(self.weight,))  # type: ignore[return-value]
