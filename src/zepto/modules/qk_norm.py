"""Q/K RMSNorm helper for grouped-query attention."""

from __future__ import annotations

from ..core.composition import GraphTensor
from .rms_norm import RMSNorm


class QKNormRMSNorm:
    """Apply separate last-dim RMSNorm to headed Q and K (Apertus QK-Norm)."""

    def __init__(self, head_dim: int, *, eps: float = 1e-5) -> None:
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def apply(
        self,
        query: GraphTensor,
        key: GraphTensor,
    ) -> tuple[GraphTensor, GraphTensor]:
        """Normalize headed Q/K tensors of shape ``(h, S, d_h)``."""
        return self.q_norm(query), self.k_norm(key)
