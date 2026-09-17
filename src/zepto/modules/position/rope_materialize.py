"""Rotary positional-encoding cache materialization module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Concat, Cos, MatMul, Multiply, Reshape, Sin

from .rope_config import RoPEConfig


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
        cfg = config or RoPEConfig(head_dim=head_dim)
        if cfg.head_dim != head_dim:
            raise ValueError(
                f"config.head_dim ({cfg.head_dim}) must match head_dim ({head_dim})"
            )
        rotary_dim = cfg.resolved_rotary_dim
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

        self.seq_len = seq_len
        self.head_dim = head_dim
        self.config = cfg

        half = rotary_dim // 2
        self._inv_freq = Tensor(
            shape=(half,),
            semantic_type="inv_freq",
            requires_grad=False,
            persistent=True,
        )
        self._position_ids = Tensor(
            shape=(seq_len,),
            semantic_type="position_ids",
            requires_grad=False,
        )
        self._attention_scaling = Tensor(
            shape=(1,),
            semantic_type="attention_scaling",
            requires_grad=False,
        )

    def forward(self) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        half = self.config.resolved_rotary_dim // 2
        position_col = Reshape(shape=(self.seq_len, 1))(self._position_ids)
        inv_row = Reshape(shape=(1, half))(self._inv_freq)
        freqs = MatMul()(position_col, inv_row)
        emb = Concat(axis=1, input_count=2)(freqs, freqs)
        cos_cache = Multiply()(Cos()(emb), self._attention_scaling)
        sin_cache = Multiply()(Sin()(emb), self._attention_scaling)
        return cos_cache, sin_cache  # type: ignore[return-value]
