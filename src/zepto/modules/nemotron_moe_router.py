"""Nemotron grouped sigmoid MoE router."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import (
    Divide,
    Gather,
    LinearMatMul,
    Multiply,
    ParameterBias,
    ReduceSum,
    Reshape,
    Sigmoid,
    TopK,
)


class NemotronMoERouter(Module):
    """Grouped top-k sigmoid router matching ``NemotronHTopkRouter``."""

    module_kind = "NemotronMoERouter"

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        top_k: int = 6,
        num_groups: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
    ) -> None:
        super().__init__()
        if num_experts % num_groups != 0:
            raise ValueError("num_experts must be divisible by num_groups")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_groups = num_groups
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.experts_per_group = num_experts // num_groups
        self.weight = Parameter(shape=(hidden_size, num_experts), semantic_type="weight")
        self.e_score_correction_bias = Parameter(
            shape=(num_experts,), semantic_type="bias"
        )
        self._scale = Tensor(
            shape=(1,), semantic_type="routed_scaling_factor", requires_grad=False
        )
        if num_groups > 1:
            self._group_mask = Tensor(
                shape=(1, num_experts),
                semantic_type="synthetic_group_mask",
                requires_grad=False,
            )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        router_logits = LinearMatMul()(  # type: ignore[call-arg]
            hidden_states, parameters=(self.weight,)
        )
        scores = Sigmoid()(router_logits)
        scores_for_choice = ParameterBias()(  # type: ignore[call-arg]
            scores, parameters=(self.e_score_correction_bias,)
        )

        token_count = scores_for_choice.shape[0]
        grouped = Reshape(
            shape=(token_count, self.num_groups, self.experts_per_group)
        )(scores_for_choice)
        top2_values, _ = TopK(k=2, dim=-1)(grouped)
        group_scores = ReduceSum(axis=-1)(top2_values)
        _group_vals, group_idx = TopK(k=self.topk_group, dim=-1)(group_scores)

        if self.num_groups > 1:
            masked_scores = Multiply()(scores_for_choice, self._group_mask)  # type: ignore[call-arg]
        else:
            masked_scores = scores_for_choice

        _top_vals, topk_indices = TopK(k=self.top_k, dim=-1)(masked_scores)
        topk_weights = Gather(axis=-1)(scores, topk_indices)
        if self.norm_topk_prob:
            denom = ReduceSum(axis=-1, keepdim=True)(topk_weights)
            topk_weights = Divide()(topk_weights, denom)  # type: ignore[call-arg]
        topk_weights = Multiply()(topk_weights, self._scale)  # type: ignore[return-value]
        del group_idx
        return router_logits, topk_weights, topk_indices  # type: ignore[return-value]


__all__ = ["NemotronMoERouter"]
