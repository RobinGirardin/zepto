"""Materialized causal mask module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import materialized_causal_mask


class MaterializedCausalMask(Module):
    """Model-level shared additive causal mask ``(1, S, S)``."""

    module_kind = "MaterializedCausalMask"

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len

    def forward(self) -> GraphTensor:
        return materialized_causal_mask(self.seq_len)
