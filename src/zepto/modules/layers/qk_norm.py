"""Q/K RMSNorm helper for grouped-query attention."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from .rms_norm import RMSNorm


class QKNormRMSNorm(Module):
    """Apply separate last-dim RMSNorm to headed Q and K (Apertus QK-Norm)."""

    module_kind = "QKNormRMSNorm"

    def __init__(self, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize headed Q/K tensors ``(h, S, d_h)`` or ``(B, h, S, d_h)``."""
        return self.q_norm(query), self.k_norm(key)  # type: ignore[return-value]
