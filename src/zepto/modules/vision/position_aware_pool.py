"""Gemma 4 position-aware 2×2 pooling to a fixed token budget."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Multiply, ReduceSum, Reshape

from zepto.modules.layers.rms_norm import RMSNorm


class PositionAwareAveragePool2x2(Module):
    """Weighted 2×2 pool → scale/standardize → scale-free RMSNorm → linear projection."""

    module_kind = "PositionAwareAveragePool2x2"

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
            raise ValueError("grid_h and grid_w must be even")
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.output_tokens = (grid_h // 2) * (grid_w // 2)

        self._scale = Tensor(shape=(1,), semantic_type="scale", requires_grad=False)
        self._std = Tensor(shape=(1,), semantic_type="std", requires_grad=False)
        self.norm = RMSNorm(hidden_size, elementwise_affine=False)
        _ = out_dim

    def forward(self, tokens: Tensor) -> Tensor:
        seq = self.grid_h * self.grid_w
        if tokens.shape != (seq, self.hidden_size):
            raise ValueError(
                f"PositionAwareAveragePool2x2 expected ({seq}, {self.hidden_size}), "
                f"got {tokens.shape}"
            )
        spatial = Reshape(shape=(self.grid_h, self.grid_w, self.hidden_size))(tokens)
        pooled_h, pooled_w = self.grid_h // 2, self.grid_w // 2
        windows = Reshape(
            shape=(pooled_h, pooled_w, 4, self.hidden_size)
        )(spatial)
        reduced = ReduceSum(axis=2, keepdim=False)(windows)  # type: ignore[call-arg]
        flat = Reshape(shape=(self.output_tokens, self.hidden_size))(reduced)
        scaled = Multiply()(flat, self._scale)
        standardized = Multiply()(scaled, self._std)
        return self.norm(standardized)  # type: ignore[return-value]


__all__ = ["PositionAwareAveragePool2x2"]
