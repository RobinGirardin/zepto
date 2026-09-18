"""Laguna sparse MoE block with shared SwiGLU expert."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Multiply, Reshape

from .laguna_expert_pool import LagunaExpertPool
from .laguna_moe_router import LagunaMoERouter
from zepto.modules.moe.moe_routing import UniformRoutingProfile
from zepto.modules.ffn.swiglu import SwiGLU


class LagunaSparseMoEBlock(Module):
    """Sparse MoE FFN matching ``LagunaSparseMoeBlock``."""

    module_kind = "LagunaSparseMoEBlock"

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        *,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        routing_profile: UniformRoutingProfile | None = None,
        expert_indices: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.routed_scaling_factor = routed_scaling_factor
        self.router = LagunaMoERouter(hidden_size, num_experts, top_k=top_k)
        self.expert_pool = LagunaExpertPool(
            hidden_size,
            moe_intermediate_size,
            num_experts,
            top_k=top_k,
            routing_profile=routing_profile,
            expert_indices=expert_indices,
        )
        self.shared_expert = SwiGLU(hidden_size, shared_intermediate_size)
        self._scale = Tensor(
            shape=(1,), semantic_type="routed_scaling_factor", requires_grad=False
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
                "LagunaSparseMoEBlock expects rank-2 (S, H) or rank-3 (B, S, H)"
            )
        shared_out = self.shared_expert(flat)
        _logits, weights, indices = self.router(flat)
        routed_out = self.expert_pool(flat, indices, weights)
        scaled = Multiply()(routed_out, self._scale)  # type: ignore[call-arg]
        out = Add()(scaled, shared_out)  # type: ignore[call-arg]
        if rank == 3:
            return Reshape(shape=orig_shape)(out)  # type: ignore[return-value]
        return out  # type: ignore[return-value]


__all__ = ["LagunaSparseMoEBlock"]
