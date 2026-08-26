"""Language-model head module."""

from __future__ import annotations

from ..core.composition import GraphParameter, GraphTensor, Module
from ..core.functional import linear_matmul
from ..core.metadata import ValueMetadata
from ._helpers import require_context


class LMHead(Module):
    """Untied vocabulary projection ``Linear(d → V)`` on final hidden states."""

    module_kind = "LMHead"

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        weight: GraphParameter | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weight = (
            weight
            if weight is not None
            else require_context().parameter(
                ValueMetadata(
                    (hidden_size, vocab_size),
                    semantic_type="weight",
                )
            )
        )
        if self.weight.metadata.shape != (hidden_size, vocab_size):
            raise ValueError(
                f"LMHead weight shape must be {(hidden_size, vocab_size)}, "
                f"got {self.weight.metadata.shape}"
            )

    def forward(self, hidden_states: GraphTensor) -> GraphTensor:
        if len(hidden_states.metadata.shape) != 2:
            raise ValueError(
                f"LMHead expects rank-2 hidden states (S, d), "
                f"got {hidden_states.metadata.shape}"
            )
        if hidden_states.metadata.shape[1] != self.hidden_size:
            raise ValueError(
                f"LMHead hidden dim mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.metadata.shape[1]}"
            )
        return linear_matmul(hidden_states, self.weight)
