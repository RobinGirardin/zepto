"""Fused linear + cross-entropy module (Liger FLCE identity chain)."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.modules._internal._batch import (
    batch_seq_dims,
    expect_labels_for_hidden,
    expect_sequence_hidden_states,
)
from zepto.semantic import (
    Divide,
    Exp,
    Gather,
    LinearMatMul,
    Log,
    Multiply,
    ReduceSum,
    Reshape,
)


class FusedLinearCrossEntropy(Module):
    """LM-head training path: ``Linear(d→V)`` + vocab softmax + CE loss.

    Decomposed eager chain matching unfused ``Linear + CrossEntropyLoss``.
    Fused by ``region/linear_ce`` when ``requested_capabilities={'fused'}``.
    """

    module_kind = "FusedLinearCrossEntropy"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        weight: Parameter | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

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

    def forward(self, hidden_states: Tensor, labels: Tensor) -> Tensor:
        expect_sequence_hidden_states(hidden_states, self.hidden_size)
        expect_labels_for_hidden(labels, hidden_states)

        logits = LinearMatMul()(
            hidden_states,
            parameters=(self.weight,),
        )
        rank = len(hidden_states.shape)
        if rank == 3:
            batch, seq_len = batch_seq_dims(hidden_states)
            token_count = batch * seq_len
            logits = Reshape(shape=(token_count, self.vocab_size))(logits)
            labels = Reshape(shape=(token_count,))(labels)

        exp_logits = Exp()(logits)
        sum_exp = ReduceSum(axis=-1, keepdim=True)(exp_logits)
        probs = Divide()(exp_logits, sum_exp)
        selected = Gather(axis=-1)(probs, labels)
        log_prob = Log()(selected)
        neg_log = Multiply()(log_prob, self._neg_one)
        return ReduceSum(axis=0, keepdim=False)(neg_log)  # type: ignore[return-value]
