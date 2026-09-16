"""Shared MoE routing profile and index synthesis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zepto.compose import Module, Tensor


@dataclass(frozen=True)
class UniformRoutingProfile:
    """Assume uniform load: each expert receives ``tokens_per_expert`` tokens."""

    tokens_per_expert: int


def tokens_per_expert_uniform(seq_len: int, top_k: int, num_experts: int) -> int:
    """Compute uniform per-expert token count for estimation."""
    return math.ceil(seq_len * top_k / num_experts)


def default_uniform_profile(
    seq_len: int, top_k: int, num_experts: int
) -> UniformRoutingProfile:
    """Build the default v1 uniform routing profile."""
    return UniformRoutingProfile(
        tokens_per_expert=tokens_per_expert_uniform(seq_len, top_k, num_experts)
    )


def synthesize_expert_indices(
    module: Module,
    num_experts: int,
    tokens_per_expert: int,
) -> tuple[Tensor, ...]:
    """Create and register synthetic rank-1 index tensors for shape inference."""
    indices: list[Tensor] = []
    for expert_id in range(num_experts):
        idx = Tensor(
            shape=(tokens_per_expert,),
            semantic_type="synthetic_routing_index",
            requires_grad=False,
        )
        name = f"_routing_idx_{expert_id}"
        setattr(module, name, idx)
        indices.append(getattr(module, name))
    return tuple(indices)


def zero_accumulator_like(hidden_states: Tensor, zero: Tensor) -> Tensor:
    """Return a zero tensor matching ``hidden_states`` shape for scatter init."""
    from zepto.semantic import Multiply

    return Multiply()(hidden_states, zero)  # type: ignore[return-value]


__all__ = [
    "UniformRoutingProfile",
    "default_uniform_profile",
    "synthesize_expert_indices",
    "tokens_per_expert_uniform",
    "zero_accumulator_like",
]
