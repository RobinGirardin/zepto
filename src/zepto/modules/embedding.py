"""Token embedding module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.metadata import ValueMetadata
from ..core.operation import EmbeddingLookup
from ._helpers import require_context


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
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if "weight" not in self._parameters:
            self.weight = require_context().parameter(
                ValueMetadata(
                    (self.hidden_size, self.vocab_size),
                    semantic_type="weight",
                )
            )
        self._initialized = True

    def forward(self, token_ids: GraphTensor) -> GraphTensor:
        self._ensure_initialized()
        if len(token_ids.metadata.shape) != 1:
            raise ValueError(
                f"Embedding token_ids must be rank-1 (S,), got {token_ids.metadata.shape}"
            )
        return require_context().apply(  # type: ignore[return-value]
            EmbeddingLookup(),
            token_ids,
            parameters=(self.weight,),
        )
