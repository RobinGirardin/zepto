"""Apertus training model with fused linear cross-entropy loss."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from .apertus import Apertus
from .fused_linear_cross_entropy import FusedLinearCrossEntropy


class ApertusForCausalLM(Module):
    """Training wrapper: backbone hidden states + shared-weight fused CE loss."""

    module_kind = "ApertusForCausalLM"

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        vocab_size: int,
        seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = Apertus(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            seq_len=seq_len,
            head_dim=head_dim,
        )
        self.loss_head = FusedLinearCrossEntropy(
            hidden_size,
            vocab_size,
            weight=self.backbone.lm_head.weight,
        )

    def forward(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        hidden = self.backbone.forward_hidden(token_ids)
        return self.loss_head(hidden, labels)  # type: ignore[return-value]
