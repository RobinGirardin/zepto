"""Qwen3-VL 2×2 spatial patch merger."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Reshape

from .layer_norm import LayerNorm
from .linear import Linear


class PatchMerger2x2(Module):
    """Group four spatial tokens → LayerNorm → MLP → ``d_out`` (Qwen vision merger)."""

    module_kind = "PatchMerger2x2"

    def __init__(
        self,
        hidden_size: int,
        out_dim: int,
        *,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        if grid_h % 2 or grid_w % 2:
            raise ValueError("grid_h and grid_w must be even for 2×2 merge")
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.norm = LayerNorm(hidden_size * 4)
        self.fc1 = Linear(hidden_size * 4, out_dim)
        self.fc2 = Linear(out_dim, out_dim)

    @property
    def output_seq_len(self) -> int:
        return (self.grid_h // 2) * (self.grid_w // 2)

    def forward(self, tokens: Tensor) -> Tensor:
        seq = self.grid_h * self.grid_w
        if tokens.shape != (seq, self.hidden_size):
            raise ValueError(
                f"PatchMerger2x2 expected ({seq}, {self.hidden_size}), got {tokens.shape}"
            )
        merged_h, merged_w = self.grid_h // 2, self.grid_w // 2
        grouped = Reshape(shape=(merged_h, merged_w, self.hidden_size * 4))(tokens)
        flat = Reshape(shape=(self.output_seq_len, self.hidden_size * 4))(grouped)
        normed = self.norm(flat)
        hidden = self.fc1(normed)
        return self.fc2(hidden)  # type: ignore[return-value]


__all__ = ["PatchMerger2x2"]
