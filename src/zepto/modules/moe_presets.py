"""Checkpoint-faithful MoE block preset factories."""

from __future__ import annotations

from .gpt_oss_moe_block import GptOssMoEBlock
from .laguna_sparse_moe_block import LagunaSparseMoEBlock
from .moe_routing import UniformRoutingProfile, default_uniform_profile
from .nemotron_moe_block import NemotronMoEBlock


def gpt_oss_moe_block(
    *,
    hidden_size: int = 2880,
    intermediate_size: int = 2880,
    num_experts: int = 32,
    num_experts_per_tok: int = 4,
    seq_len: int = 128,
    routing_profile: UniformRoutingProfile | None = None,
) -> GptOssMoEBlock:
    """GPT-OSS 20B sparse MoE block preset."""
    profile = routing_profile or default_uniform_profile(
        seq_len, num_experts_per_tok, num_experts
    )
    return GptOssMoEBlock(
        hidden_size,
        intermediate_size,
        num_experts,
        top_k=num_experts_per_tok,
        routing_profile=profile,
    )


def laguna_sparse_moe_block(
    *,
    hidden_size: int = 2048,
    moe_intermediate_size: int = 512,
    shared_intermediate_size: int = 512,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    routed_scaling_factor: float = 2.5,
    seq_len: int = 128,
    routing_profile: UniformRoutingProfile | None = None,
) -> LagunaSparseMoEBlock:
    """Laguna XS 2.1 sparse MoE block preset."""
    profile = routing_profile or default_uniform_profile(
        seq_len, num_experts_per_tok, num_experts
    )
    return LagunaSparseMoEBlock(
        hidden_size,
        moe_intermediate_size,
        shared_intermediate_size,
        num_experts,
        top_k=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        routing_profile=profile,
    )


def nemotron_moe_block(
    *,
    hidden_size: int = 2688,
    moe_intermediate_size: int = 1856,
    shared_intermediate_size: int = 3712,
    num_experts: int = 128,
    num_experts_per_tok: int = 6,
    num_groups: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
    seq_len: int = 128,
    routing_profile: UniformRoutingProfile | None = None,
) -> NemotronMoEBlock:
    """Nemotron 3.5 Lightning MoE block preset."""
    profile = routing_profile or default_uniform_profile(
        seq_len, num_experts_per_tok, num_experts
    )
    return NemotronMoEBlock(
        hidden_size,
        moe_intermediate_size,
        shared_intermediate_size,
        num_experts,
        top_k=num_experts_per_tok,
        num_groups=num_groups,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        routing_profile=profile,
    )


__all__ = [
    "gpt_oss_moe_block",
    "laguna_sparse_moe_block",
    "nemotron_moe_block",
]
