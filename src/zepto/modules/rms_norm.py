"""Root-mean-square normalization module boundary."""

from ..core.composition import GraphTensor, Module
from ..core.functional import identity


class RMSNorm(Module):
    """Reusable normalization composition boundary."""

    def __init__(self, normalized_shape: int, epsilon: float = 1e-5) -> None:
        """Initialize an RMS-normalization configuration.

        Args:
            normalized_shape: Size of the normalized feature dimension.
            epsilon: Numerical-stability constant used by normalization.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

    def forward(self, value: GraphTensor) -> GraphTensor:
        """Compose the current placeholder RMS-normalization behavior."""
        return identity(value)
