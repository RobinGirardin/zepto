"""Nemotron MoE block with shared SquaredReluFFN expert."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Reshape

from .moe_routing import UniformRoutingProfile
from .nemotron_expert_pool import NemotronExpertPool
from .nemotron_moe_router import NemotronMoERouter
from .squared_relu_ffn import SquaredReluFFN


class NemotronMoEBlock(Module):
    """Sparse MoE FFN matching ``NemotronHMoE`` (without latent projections)."""

    module_kind = "NemotronMoEBlock"

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        *,
        top_k: int = 6,
        num_groups: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        routing_profile: UniformRoutingProfile | None = None,
        expert_indices: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.router = NemotronMoERouter(
            hidden_size,
            num_experts,
            top_k=top_k,
            num_groups=num_groups,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.expert_pool = NemotronExpertPool(
            hidden_size,
            moe_intermediate_size,
            num_experts,
            top_k=top_k,
            routing_profile=routing_profile,
            expert_indices=expert_indices,
        )
        self.shared_expert = SquaredReluFFN(hidden_size, shared_intermediate_size)

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
                "NemotronMoEBlock expects rank-2 (S, H) or rank-3 (B, S, H)"
            )
        residuals = flat
        _logits, weights, indices = self.router(flat)
        routed_out = self.expert_pool(flat, indices, weights)
        shared_out = self.shared_expert(residuals)
        out = Add()(routed_out, shared_out)  # type: ignore[call-arg]
        if rank == 3:
            return Reshape(shape=orig_shape)(out)  # type: ignore[return-value]
        return out  # type: ignore[return-value]


__all__ = ["NemotronMoEBlock"]
