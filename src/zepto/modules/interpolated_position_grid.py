"""Bilinear-interpolated 2-D position grid for vision encoders."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import BilinearResize2D, Reshape


class InterpolatedPositionGrid(Module):
    """Resize a learned 2-D position grid to a target patch resolution.

    Returns flattened positions ``(out_h * out_w, embed_dim)``.
    """

    module_kind = "InterpolatedPositionGrid"

    def __init__(
        self,
        src_h: int,
        src_w: int,
        embed_dim: int,
        *,
        out_h: int,
        out_w: int,
    ) -> None:
        super().__init__()
        if min(src_h, src_w, embed_dim, out_h, out_w) <= 0:
            raise ValueError("all grid and embed dimensions must be positive")

        self.src_h = src_h
        self.src_w = src_w
        self.embed_dim = embed_dim
        self.out_h = out_h
        self.out_w = out_w

        self.position_weight = Parameter(
            shape=(src_h, src_w, embed_dim),
            semantic_type="position_embedding",
        )

    def forward(self) -> Tensor:
        resized = BilinearResize2D(out_h=self.out_h, out_w=self.out_w)(
            parameters=(self.position_weight,),
        )
        return Reshape(shape=(self.out_h * self.out_w, self.embed_dim))(resized)  # type: ignore[return-value]
