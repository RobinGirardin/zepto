"""Axial 2-D RoPE cache materialization for vision encoders."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Concat, Cos, MatMul, Multiply, RepeatKV, Reshape, Sin

from .rope_config import RoPEConfig


def _materialize_1d(
    positions: Tensor,
    inv_freq: Tensor,
    seq_len: int,
    axis_dim: int,
    attention_scaling: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build cos/sin caches of width ``axis_dim`` from 1-D positions."""
    freq_count = axis_dim // 2
    position_col = Reshape(shape=(seq_len, 1))(positions)
    inv_row = Reshape(shape=(1, freq_count))(inv_freq)
    freqs = MatMul()(position_col, inv_row)
    emb = Concat(axis=1, input_count=2)(freqs, freqs)
    cos_cache = Multiply()(Cos()(emb), attention_scaling)
    sin_cache = Multiply()(Sin()(emb), attention_scaling)
    return cos_cache, sin_cache


class AxialRoPEMaterialize(Module):
    """Materialize 2-D axial RoPE ``cos``/``sin`` caches for a ``grid_h × grid_w`` patch grid.

    Height positions encode the first ``head_dim // 2`` channels; width positions
    encode the second half. Each axis cache is broadcast across the other axis,
    concatenated on the channel axis, and flattened to ``(grid_h * grid_w, head_dim)``.
    """

    module_kind = "AxialRoPEMaterialize"

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        head_dim: int,
        *,
        config: RoPEConfig | None = None,
    ) -> None:
        super().__init__()
        if grid_h <= 0 or grid_w <= 0 or head_dim <= 0:
            raise ValueError("grid_h, grid_w, and head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for axial RoPE, got {head_dim}")
        cfg = config or RoPEConfig(head_dim=head_dim)
        if cfg.head_dim != head_dim:
            raise ValueError(
                f"config.head_dim ({cfg.head_dim}) must match head_dim ({head_dim})"
            )

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.head_dim = head_dim
        self.config = cfg
        axis_dim = head_dim // 2
        freq_count = axis_dim // 2

        self._inv_freq_h = Tensor(
            shape=(freq_count,),
            semantic_type="inv_freq",
            requires_grad=False,
            persistent=True,
        )
        self._inv_freq_w = Tensor(
            shape=(freq_count,),
            semantic_type="inv_freq",
            requires_grad=False,
            persistent=True,
        )
        self._position_ids_h = Tensor(
            shape=(grid_h,),
            semantic_type="position_ids",
            requires_grad=False,
        )
        self._position_ids_w = Tensor(
            shape=(grid_w,),
            semantic_type="position_ids",
            requires_grad=False,
        )
        self._attention_scaling = Tensor(
            shape=(1,),
            semantic_type="attention_scaling",
            requires_grad=False,
        )

    def forward(self) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        axis_dim = self.head_dim // 2
        cos_h, sin_h = _materialize_1d(
            self._position_ids_h,
            self._inv_freq_h,
            self.grid_h,
            axis_dim,
            self._attention_scaling,
        )
        cos_w, sin_w = _materialize_1d(
            self._position_ids_w,
            self._inv_freq_w,
            self.grid_w,
            axis_dim,
            self._attention_scaling,
        )

        cos_h_grid = RepeatKV(n_rep=self.grid_w, axis=1)(
            Reshape(shape=(self.grid_h, 1, axis_dim))(cos_h)
        )
        sin_h_grid = RepeatKV(n_rep=self.grid_w, axis=1)(
            Reshape(shape=(self.grid_h, 1, axis_dim))(sin_h)
        )
        cos_w_grid = RepeatKV(n_rep=self.grid_h, axis=0)(
            Reshape(shape=(1, self.grid_w, axis_dim))(cos_w)
        )
        sin_w_grid = RepeatKV(n_rep=self.grid_h, axis=0)(
            Reshape(shape=(1, self.grid_w, axis_dim))(sin_w)
        )

        cos_grid = Concat(axis=2, input_count=2)(cos_h_grid, cos_w_grid)
        sin_grid = Concat(axis=2, input_count=2)(sin_h_grid, sin_w_grid)

        flat_size = self.grid_h * self.grid_w
        cos_flat = Reshape(shape=(flat_size, self.head_dim))(cos_grid)
        sin_flat = Reshape(shape=(flat_size, self.head_dim))(sin_grid)
        return cos_flat, sin_flat  # type: ignore[return-value]
