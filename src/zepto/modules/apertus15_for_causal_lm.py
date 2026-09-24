"""Apertus 1.5 text-only training wrapper with fused linear cross-entropy."""

from __future__ import annotations

from zepto.compose import Module, Tensor

from .models.apertus15 import Apertus15
from .output.fused_linear_cross_entropy import FusedLinearCrossEntropy


class Apertus15ForCausalLM(Module):
    """Training wrapper: backbone hidden + shared-weight fused CE on pruned vocab.

    ::

        Apertus15ForCausalLM(
            *,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            num_kv_heads: int,
            num_layers: int,
            vocab_size: int,            # input Embedding rows
            output_vocab_size: int,     # LMHead / fused-CE columns
            seq_len: int,
            head_dim: int | None = None,
        ) -> Module

        Apertus15ForCausalLM.forward(token_ids, labels) -> scalar CE
    """

    module_kind = "Apertus15ForCausalLM"

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        vocab_size: int,
        output_vocab_size: int,
        seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = Apertus15(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            output_vocab_size=output_vocab_size,
            seq_len=seq_len,
            head_dim=head_dim,
        )
        self.loss_head = FusedLinearCrossEntropy(
            hidden_size,
            output_vocab_size,
            weight=self.backbone.lm_head.weight,
        )

    def forward(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        hidden = self.backbone.forward_hidden(token_ids)
        return self.loss_head(hidden, labels)  # type: ignore[return-value]


__all__ = ["Apertus15ForCausalLM"]
