"""Learned 2-D position embedding for vision encoders."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import EmbeddingLookup


class LearnedPositionEmbedding2D(Module):
    """Gather learned position vectors for a ``grid_h × grid_w`` patch grid.

    Returns flattened positions ``(grid_h * grid_w, embed_dim)`` with zero FLOPs.
    """

    module_kind = "LearnedPositionEmbedding2D"

    def __init__(
        self,
        max_grid_h: int,
        max_grid_w: int,
        embed_dim: int,
        *,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        if min(max_grid_h, max_grid_w, embed_dim, grid_h, grid_w) <= 0:
            raise ValueError("all grid and embed dimensions must be positive")
        if grid_h > max_grid_h or grid_w > max_grid_w:
            raise ValueError("grid_h/grid_w must not exceed max_grid_h/max_grid_w")

        self.max_grid_h = max_grid_h
        self.max_grid_w = max_grid_w
        self.embed_dim = embed_dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.position_weight = Parameter(
            shape=(embed_dim, max_grid_h * max_grid_w),
            semantic_type="weight",
        )
        flat_count = grid_h * grid_w
        self._flat_indices = Tensor(
            shape=(flat_count,),
            semantic_type="position_indices",
            requires_grad=False,
        )

    def forward(self) -> Tensor:
        return EmbeddingLookup()(
            self._flat_indices,
            parameters=(self.position_weight,),
        )  # type: ignore[return-value]
