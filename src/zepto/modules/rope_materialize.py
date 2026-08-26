"""Rotary positional-encoding cache materialization module."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.composition import GraphCompositionContext, GraphTensor, Module
from ..core.functional import concat, cos, matmul, multiply, reshape, scalar_input, sin
from ..core.metadata import ValueMetadata
from ._helpers import require_context


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """RoPE frequency/scaling knobs (init-only; shapes unchanged at runtime)."""

    attention_scaling: float = 1.0


class RoPEMaterialize(Module):
    """Materialize forward-lived ``cos``/``sin`` caches from ``inv_freq`` buffers.

    Matches HuggingFace ``LlamaRotaryEmbedding`` eager rebuild:

    ``freqs = outer(position_ids, inv_freq) → concat → cos/sin × attention_scaling``.
    """

    module_kind = "RoPEMaterialize"

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        *,
        config: RoPEConfig | None = None,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or head_dim <= 0:
            raise ValueError("seq_len and head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.config = config or RoPEConfig()

        ctx = require_context()
        half = head_dim // 2
        self._inv_freq = ctx.input(
            ValueMetadata(
                (half,),
                semantic_type="inv_freq",
                requires_grad=False,
                persistent=True,
            )
        )
        self._position_ids = ctx.input(
            ValueMetadata(
                (seq_len,),
                semantic_type="position_ids",
                requires_grad=False,
            )
        )
        self._attention_scaling = scalar_input(semantic_type="attention_scaling")

    def forward(self) -> tuple[GraphTensor, GraphTensor]:
        half = self.head_dim // 2
        position_col = reshape(self._position_ids, (self.seq_len, 1))
        inv_row = reshape(self._inv_freq, (1, half))
        freqs = matmul(position_col, inv_row)
        emb = concat(freqs, freqs, axis=1)
        cos_cache = multiply(cos(emb), self._attention_scaling)
        sin_cache = multiply(sin(emb), self._attention_scaling)
        return cos_cache, sin_cache
