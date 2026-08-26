"""Linear layer module."""

from ..core.composition import (
    GraphCompositionContext,
    GraphTensor,
    Module,
)
from ..core.functional import linear_matmul
from ..core.metadata import ValueMetadata


class Linear(Module):
    """Affine transform via activation × weight parameter (weight-only)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        ctx = GraphCompositionContext.current()
        if ctx is None:
            raise RuntimeError("Linear requires an active GraphCompositionContext")

        self.weight = ctx.parameter(
            ValueMetadata(
                (in_features, out_features),
                semantic_type="weight",
            )
        )

    def forward(self, x: GraphTensor) -> GraphTensor:
        return linear_matmul(x, self.weight)
