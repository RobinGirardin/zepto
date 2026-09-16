"""GPT-OSS routed expert pool with gather/scatter dispatch."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Gather, Multiply, ScatterAdd

from .gpt_oss_expert import GptOssExpert
from .moe_routing import UniformRoutingProfile, zero_accumulator_like


class GptOssExpertPool(Module):
    """Unrolled per-expert dispatch via ``Gather`` / ``ScatterAdd``."""

    module_kind = "GptOssExpertPool"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        top_k: int,
        routing_profile: UniformRoutingProfile | None = None,
        expert_indices: tuple[Tensor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        for expert_id in range(num_experts):
            setattr(
                self,
                f"expert_{expert_id}",
                GptOssExpert(hidden_size, intermediate_size),
            )
        if expert_indices is not None:
            if len(expert_indices) != num_experts:
                raise ValueError("expert_indices length must match num_experts")
            self._expert_indices = expert_indices
        elif routing_profile is not None:
            from .moe_routing import synthesize_expert_indices

            self._expert_indices = synthesize_expert_indices(
                self,
                num_experts,
                routing_profile.tokens_per_expert,
            )
        else:
            raise ValueError("routing_profile or expert_indices required")
        self._uniform_weight = Tensor(
            shape=(self._expert_indices[0].shape[0], 1),
            semantic_type="routing_weight",
            requires_grad=False,
        )
        self._zero = Tensor(
            shape=(1,), semantic_type="constant_zero", requires_grad=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        top_indices: Tensor,
        router_weights: Tensor,
    ) -> Tensor:
        del top_indices, router_weights
        acc = zero_accumulator_like(hidden_states, self._zero)
        for expert_id in range(self.num_experts):
            expert = getattr(self, f"expert_{expert_id}")
            indices = self._expert_indices[expert_id]
            gathered = Gather(axis=0)(hidden_states, indices)
            expert_out = expert(gathered)
            weighted = Multiply()(expert_out, self._uniform_weight)  # type: ignore[call-arg]
            acc = ScatterAdd(axis=0)(acc, indices, weighted)  # type: ignore[call-arg,assignment]
        return acc  # type: ignore[return-value]


__all__ = ["GptOssExpertPool"]
