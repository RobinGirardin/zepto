"""Fuse text embeddings with projected visual tokens at placeholder indices."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import ScatterUpdate

from .embedding import Embedding


class MultimodalSequenceBuilder(Module):
    """Text ``Embedding`` lookup followed by row ``ScatterUpdate`` fusion.

    Effective downstream sequence length for costing:
    ``S' = S_text - n_placeholders + S_v`` when placeholders are replaced by
    variable-length visual token runs (document in caller / ``EstimationContext``).
    """

    module_kind = "MultimodalSequenceBuilder"

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = Embedding(hidden_size, vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        visual_tokens: Tensor,
        placeholder_indices: Tensor,
    ) -> Tensor:
        text_embeds = self.embedding(token_ids)
        return ScatterUpdate(axis=0)(
            text_embeds,
            placeholder_indices,
            visual_tokens,
        )  # type: ignore[return-value]


__all__ = ["MultimodalSequenceBuilder"]
