"""GPT-OSS MoE block (router + expert pool)."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Reshape

from .gpt_oss_expert_pool import GptOssExpertPool
from .gpt_oss_moe_router import GptOssMoERouter
from .moe_routing import UniformRoutingProfile


class GptOssMoEBlock(Module):
    """Sparse MoE FFN matching ``GptOssMLP`` eager decomposition."""

    module_kind = "GptOssMoEBlock"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        top_k: int = 4,
        routing_profile: UniformRoutingProfile | None = None,
        expert_indices: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.router = GptOssMoERouter(hidden_size, num_experts, top_k=top_k)
        self.expert_pool = GptOssExpertPool(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k=top_k,
            routing_profile=routing_profile,
            expert_indices=expert_indices,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        rank = len(hidden_states.shape)
        if rank == 2:
            flat = hidden_states
            orig_shape = hidden_states.shape
        elif rank == 3:
            batch, seq_len, hidden = hidden_states.shape
            flat = Reshape(shape=(batch * seq_len, hidden))(hidden_states)
            orig_shape = hidden_states.shape
        else:
            raise ValueError(
                "GptOssMoEBlock expects rank-2 (S, H) or rank-3 (B, S, H) input"
            )
        _logits, weights, indices = self.router(flat)
        out = self.expert_pool(flat, indices, weights)
        if rank == 3:
            return Reshape(shape=orig_shape)(out)  # type: ignore[return-value]
        return out  # type: ignore[return-value]


__all__ = ["GptOssMoEBlock"]
