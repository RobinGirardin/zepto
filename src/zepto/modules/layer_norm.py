"""Layer-normalization module boundary."""

from ..core.composition import GraphTensor, Module
from ..core.functional import identity


class LayerNorm(Module):
    """Placeholder module boundary for a primitive-composed layer norm."""

    def __init__(self, normalized_shape: int, epsilon: float = 1e-5) -> None:
        """Initialize a layer-normalization configuration.

        Args:
            normalized_shape: Size of the normalized feature dimension.
            epsilon: Numerical-stability constant used by normalization.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

    def forward(self, value: GraphTensor) -> GraphTensor:
        """Compose the current placeholder layer-normalization behavior."""
        return identity(value)
