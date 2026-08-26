"""Linear layer module."""

from zepto.compose import Compose, Module, Parameter, Tensor
from zepto.semantic import LinearMatMul


class Linear(Module):
    """Affine transform via activation × weight parameter (weight-only)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        ctx = Compose.current()
        if ctx is None:
            raise RuntimeError("Linear requires an active Compose context")

        weight = ctx.parameter(
            Parameter(
                shape=(in_features, out_features),
                semantic_type="weight",
            )
        )
        self.register_parameter("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        return LinearMatMul()(x, parameters=(self._parameters["weight"],))  # type: ignore[return-value]
