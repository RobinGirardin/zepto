"""Sliding-window causal mask module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import (
    MaterializedSlidingWindowCausalMask as MaterializedSlidingWindowCausalMaskOp,
)


class SlidingWindowCausalMask(Module):
    """Model-level shared additive sliding-window causal mask ``(1, S, S)``."""

    module_kind = "SlidingWindowCausalMask"

    def __init__(self, seq_len: int, window_size: int) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size > seq_len:
            raise ValueError("window_size must not exceed seq_len")
        self.seq_len = seq_len
        self.window_size = window_size

    def forward(self) -> Tensor:  # type: ignore[override]
        return MaterializedSlidingWindowCausalMaskOp(
            self.seq_len, self.window_size
        )()  # type: ignore[return-value]


__all__ = ["SlidingWindowCausalMask"]
