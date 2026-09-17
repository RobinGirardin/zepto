"""Multimodal sectioned RoPE (mRoPE) cache materialization."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Concat, Cos, MatMul, Multiply, Reshape, Sin, Split

from .rope_config import RoPEConfig


class MultimodalRoPEMaterialize(Module):
    """Materialize mRoPE ``cos``/``sin`` caches from multi-axis ``position_ids``.

    For each section ``s``, ``freqs_s = matmul(position_ids[s], inv_freq_slice_s)``.
    Section frequency rows are concatenated along the embedding axis (width
    ``rotary_dim // 2``), then doubled: ``emb = concat(freqs, freqs)``.
    """

    module_kind = "MultimodalRoPEMaterialize"

    def __init__(self, seq_len: int, config: RoPEConfig) -> None:
        super().__init__()
        if config.rope_type != "mrope":
            raise ValueError("MultimodalRoPEMaterialize requires rope_type='mrope'")
        if config.mrope_section is None:
            raise ValueError("MultimodalRoPEMaterialize requires mrope_section")
        if len(config.mrope_section) != config.num_sections:
            raise ValueError(
                f"mrope_section length ({len(config.mrope_section)}) must match "
                f"num_sections ({config.num_sections})"
            )
        rotary_dim = config.resolved_rotary_dim
        if seq_len <= 0 or rotary_dim <= 0:
            raise ValueError("seq_len and rotary_dim must be positive")

        self.seq_len = seq_len
        self.config = config
        self._sections = config.mrope_section

        half = rotary_dim // 2
        self._inv_freq = Tensor(
            shape=(half,),
            semantic_type="inv_freq",
            requires_grad=False,
            persistent=True,
        )
        self._position_ids = Tensor(
            shape=(config.num_sections, seq_len),
            semantic_type="position_ids",
            requires_grad=False,
        )
        self._attention_scaling = Tensor(
            shape=(1,),
            semantic_type="attention_scaling",
            requires_grad=False,
        )

    def forward(self) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        section_ids = Split(sizes=(1,) * self.config.num_sections)(self._position_ids)
        inv_slices = Split(sizes=self._sections)(self._inv_freq)

        freqs_parts: list[Tensor] = []
        for section_idx in range(self.config.num_sections):
            pos_row = Reshape(shape=(self.seq_len, 1))(section_ids[section_idx])
            inv_row = Reshape(shape=(1, self._sections[section_idx]))(
                inv_slices[section_idx]
            )
            freqs_parts.append(MatMul()(pos_row, inv_row))

        freqs = Concat(axis=1, input_count=len(freqs_parts))(*freqs_parts)
        emb = Concat(axis=1, input_count=2)(freqs, freqs)
        cos_cache = Multiply()(Cos()(emb), self._attention_scaling)
        sin_cache = Multiply()(Sin()(emb), self._attention_scaling)
        return cos_cache, sin_cache  # type: ignore[return-value]
