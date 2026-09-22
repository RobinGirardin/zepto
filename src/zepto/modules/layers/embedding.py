"""Token embedding module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.modules._internal._batch import expect_token_ids_rank
from zepto.semantic import EmbeddingLookup


class Embedding(Module):
    """Token embedding lookup with weight layout ``(d, V)`` for untied LMHead reuse.

    Matches HuggingFace eager ``nn.Embedding`` semantics via a zero-FLOP lookup op.
    """

    module_kind = "Embedding"

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.weight = Parameter(
            shape=(hidden_size, vocab_size),
            semantic_type="weight",
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        expect_token_ids_rank(token_ids)
        return EmbeddingLookup()(
            token_ids,
            parameters=(self.weight,),
        )  # type: ignore[return-value]
