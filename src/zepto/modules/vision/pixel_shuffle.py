"""Space-to-depth pixel shuffle for Muse Glimmer vision."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Reshape


class PixelShuffle2x2(Module):
    """``(H, W, C) → (H/2, W/2, 4C)`` on flattened patch tokens."""

    module_kind = "PixelShuffle2x2"

    def __init__(self, hidden_size: int, *, grid_h: int, grid_w: int) -> None:
        super().__init__()
        if grid_h % 2 or grid_w % 2:
            raise ValueError("grid_h and grid_w must be even")
        self.hidden_size = hidden_size
        self.grid_h = grid_h
        self.grid_w = grid_w

    @property
    def output_hidden_size(self) -> int:
        return self.hidden_size * 4

    @property
    def output_seq_len(self) -> int:
        return (self.grid_h // 2) * (self.grid_w // 2)

    def forward(self, tokens: Tensor) -> Tensor:
        seq = self.grid_h * self.grid_w
        if tokens.shape != (seq, self.hidden_size):
            raise ValueError(
                f"PixelShuffle2x2 expected ({seq}, {self.hidden_size}), got {tokens.shape}"
            )
        spatial = Reshape(shape=(self.grid_h, self.grid_w, self.hidden_size))(tokens)
        shuffled = Reshape(
            shape=(self.output_seq_len, self.output_hidden_size)
        )(spatial)
        return shuffled  # type: ignore[return-value]


__all__ = ["PixelShuffle2x2"]
