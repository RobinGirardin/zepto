"""Language-model head module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import LinearMatMul


class LMHead(Module):
    """Untied vocabulary projection ``Linear(d → V)`` on final hidden states."""

    module_kind = "LMHead"

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
                f"LMHead weight shape must be {(hidden_size, vocab_size)}, "
                f"got {resolved.shape}"
            )
        self.weight = resolved

    def forward(self, hidden_states: Tensor) -> Tensor:
        if len(hidden_states.shape) != 2:
            raise ValueError(
                f"LMHead expects rank-2 hidden states (S, d), "
                f"got {hidden_states.shape}"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"LMHead hidden dim mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[1]}"
            )
        return LinearMatMul()(
            hidden_states,
            parameters=(self.weight,),
        )  # type: ignore[return-value]
