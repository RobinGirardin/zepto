"""GPT-OSS MoE router (top-k + selected softmax)."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import TopK

from zepto.modules.layers.affine_linear import AffineLinear
from zepto.modules.attention.softmax import Softmax


class GptOssMoERouter(Module):
    """Top-4 softmax router matching ``GptOssTopKRouter``."""

    module_kind = "GptOssMoERouter"

    def __init__(self, hidden_size: int, num_experts: int, *, top_k: int = 4) -> None:
        super().__init__()
        if top_k <= 0 or num_experts <= 0:
            raise ValueError("top_k and num_experts must be positive")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = AffineLinear(hidden_size, num_experts, bias=True)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        router_logits = self.router(hidden_states)
        top_values, top_indices = TopK(k=self.top_k, dim=-1)(router_logits)
        router_weights = Softmax()(top_values)
        return router_logits, router_weights, top_indices  # type: ignore[return-value]


__all__ = ["GptOssMoERouter"]
