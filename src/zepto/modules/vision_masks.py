"""Non-causal vision attention mask modules."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import (
    MaterializedBidirectionalMask as MaterializedBidirectionalMaskOp,
)
from zepto.semantic import (
    MaterializedSlidingWindowBidirectionalMask as SlidingBidirectionalMaskOp,
)


class MaterializedBidirectionalMask(Module):
    """All-visible additive mask ``(1, S, S)`` for global vision layers."""

    module_kind = "MaterializedBidirectionalMask"

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len

    def forward(self) -> Tensor:  # type: ignore[override]
        return MaterializedBidirectionalMaskOp(self.seq_len)()  # type: ignore[return-value]


class SlidingWindowBidirectionalMask(Module):
    """Sliding band mask on flattened patch order (Muse local vision layers)."""

    module_kind = "SlidingWindowBidirectionalMask"

    def __init__(self, seq_len: int, window_size: int) -> None:
        super().__init__()
        if seq_len <= 0 or window_size <= 0:
            raise ValueError("seq_len and window_size must be positive")
        if window_size > seq_len:
            raise ValueError("window_size must not exceed seq_len")
        self.seq_len = seq_len
        self.window_size = window_size

    def forward(self) -> Tensor:  # type: ignore[override]
        return SlidingBidirectionalMaskOp(self.seq_len, self.window_size)()  # type: ignore[return-value]


def vision_mask_for_config(
    config: "VisionAttentionConfig", seq_len: int
) -> Module:
    from .vision_attention_config import VisionAttentionConfig

    if not isinstance(config, VisionAttentionConfig):
        raise TypeError("config must be VisionAttentionConfig")
    if config.mask == "bidirectional":
        return MaterializedBidirectionalMask(seq_len)
    if config.mask == "sliding_1d":
        assert config.window_size is not None
        return SlidingWindowBidirectionalMask(seq_len, config.window_size)
    raise ValueError(f"unsupported vision mask policy: {config.mask}")


__all__ = [
    "MaterializedBidirectionalMask",
    "SlidingWindowBidirectionalMask",
    "vision_mask_for_config",
]
