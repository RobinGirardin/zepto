"""Laguna sigmoid-routed MoE router."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Divide, Gather, LinearMatMul, ParameterBias, ReduceSum, Sigmoid, TopK


class LagunaMoERouter(Module):
    """Top-8 sigmoid router with correction-biased selection (Laguna XS)."""

    module_kind = "LagunaMoERouter"

    def __init__(self, hidden_size: int, num_experts: int, *, top_k: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = Parameter(shape=(hidden_size, num_experts), semantic_type="weight")
        self.correction_bias = Parameter(shape=(num_experts,), semantic_type="bias")

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        router_logits = LinearMatMul()(  # type: ignore[call-arg]
            hidden_states, parameters=(self.weight,)
        )
        routing_scores = Sigmoid()(router_logits)
        scores_for_selection = ParameterBias()(  # type: ignore[call-arg]
            routing_scores, parameters=(self.correction_bias,)
        )
        _top_values, selected = TopK(k=self.top_k, dim=-1)(scores_for_selection)
        routing_weights = Gather(axis=-1)(routing_scores, selected)
        weight_sum = ReduceSum(axis=-1, keepdim=True)(routing_weights)
        routing_weights = Divide()(routing_weights, weight_sum)  # type: ignore[return-value]
        return router_logits, routing_weights, selected  # type: ignore[return-value]


__all__ = ["LagunaMoERouter"]
