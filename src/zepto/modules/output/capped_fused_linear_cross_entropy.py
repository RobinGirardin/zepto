"""Fused linear + cross-entropy with optional logit soft-cap before softmax."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import (
    Divide,
    Exp,
    Gather,
    LinearMatMul,
    Log,
    Multiply,
    ReduceSum,
)

from .logit_soft_cap import LogitSoftCap, LogitSoftCapConfig


class CappedFusedLinearCrossEntropy(Module):
    """Training path: linear projection, optional soft-cap, then CE decomposition."""

    module_kind = "CappedFusedLinearCrossEntropy"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        weight: Parameter | None = None,
        soft_cap: LogitSoftCapConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.soft_cap_config = soft_cap

        resolved = weight or Parameter(
            shape=(hidden_size, vocab_size),
            semantic_type="weight",
        )
        if resolved.shape != (hidden_size, vocab_size):
            raise ValueError(
                f"weight shape must be {(hidden_size, vocab_size)}, "
                f"got {resolved.shape}"
            )
        self.weight = resolved
        self._neg_one = Tensor(
            shape=(1,), semantic_type="neg_one", requires_grad=False
        )
        self._soft_cap = LogitSoftCap(soft_cap) if soft_cap is not None else None

    def forward(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        if len(hidden_states.shape) != 2:
            raise ValueError(
                f"expected rank-2 hidden states (S, d), got {hidden_states.shape}"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[1]}"
            )
        if len(labels.shape) != 1:
            raise ValueError(
                f"expected rank-1 labels (S,), got {labels.shape}"
            )
        if labels.shape[0] != hidden_states.shape[0]:
            raise ValueError("labels and hidden states batch dim must match")

        logits = LinearMatMul()(
            hidden_states,
            parameters=(self.weight,),
        )
        if self._soft_cap is not None:
            logits = self._soft_cap(logits)  # type: ignore[assignment]

        exp_logits = Exp()(logits)
        sum_exp = ReduceSum(axis=-1, keepdim=True)(exp_logits)
        probs = Divide()(exp_logits, sum_exp)
        selected = Gather(axis=-1)(probs, labels)
        log_prob = Log()(selected)
        neg_log = Multiply()(log_prob, self._neg_one)
        return ReduceSum(axis=0, keepdim=False)(neg_log)  # type: ignore[return-value]


__all__ = ["CappedFusedLinearCrossEntropy"]
