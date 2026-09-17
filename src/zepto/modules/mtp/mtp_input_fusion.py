"""Fuse trunk hidden states with shifted token embeddings for MTP."""

from __future__ import annotations

from typing import Literal

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Concat, LinearMatMul, ParameterBias

from zepto.modules.layers.affine_linear import AffineLinear
from zepto.modules.layers.rms_norm import RMSNorm


class MtpInputFusion(Module):
    """Concat–project–norm or add–norm fusion of trunk hidden and token embed."""

    module_kind = "MtpInputFusion"

    def __init__(
        self,
        hidden_size: int,
        *,
        mode: Literal["concat_linear", "add"] = "concat_linear",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.mode = mode
        self.norm = RMSNorm(hidden_size)
        if mode == "concat_linear":
            self.project = AffineLinear(2 * hidden_size, hidden_size, bias=bias)
        else:
            self.project = None

    def forward(self, trunk_hidden: Tensor, token_embed: Tensor) -> Tensor:
        if len(trunk_hidden.shape) != 2 or len(token_embed.shape) != 2:
            raise ValueError("trunk_hidden and token_embed must be rank-2 (S, d)")
        if trunk_hidden.shape != token_embed.shape:
            raise ValueError(
                f"shape mismatch: trunk {trunk_hidden.shape} vs embed {token_embed.shape}"
            )
        if trunk_hidden.shape[1] != self.hidden_size:
            raise ValueError(
                f"expected hidden dim {self.hidden_size}, got {trunk_hidden.shape[1]}"
            )
        if self.mode == "add":
            fused = Add()(trunk_hidden, token_embed)
            return self.norm(fused)  # type: ignore[return-value]
        merged = Concat(axis=-1, input_count=2)(trunk_hidden, token_embed)  # type: ignore[call-arg]
        projected = LinearMatMul()(merged, parameters=(self.project.weight,))  # type: ignore[union-attr,call-arg]
        bias = getattr(self.project, "bias", None)  # type: ignore[union-attr]
        if bias is not None:
            projected = ParameterBias()(projected, parameters=(bias,))  # type: ignore[call-arg]
        return self.norm(projected)  # type: ignore[return-value]


__all__ = ["MtpInputFusion"]
