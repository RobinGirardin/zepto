"""Materialized causal mask module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import MaterializedCausalMask as MaterializedCausalMaskOp


class MaterializedCausalMask(Module):
    """Model-level shared additive causal mask ``(1, S, S)``."""

    module_kind = "MaterializedCausalMask"

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len

    def forward(self) -> Tensor:  # type: ignore[override]
        return MaterializedCausalMaskOp(self.seq_len)()  # type: ignore[return-value]
