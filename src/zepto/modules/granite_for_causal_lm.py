"""Granite training model with fused linear cross-entropy loss."""

from __future__ import annotations

from zepto.compose import Module, Tensor

from .models.granite import GRANITE_42_30B, Granite, GraniteConfig
from .output.fused_linear_cross_entropy import FusedLinearCrossEntropy


class GraniteForCausalLM(Module):
    """Training wrapper: backbone hidden states + shared-weight fused CE loss."""

    module_kind = "GraniteForCausalLM"

    def __init__(
        self,
        *,
        config: GraniteConfig = GRANITE_42_30B,
        seq_len: int,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = Granite(
            config=config, seq_len=seq_len, num_layers=num_layers
        )
        self.loss_head = FusedLinearCrossEntropy(
            config.hidden_size,
            config.vocab_size,
            weight=self.backbone.lm_output.weight,
        )

    def forward(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        hidden = self.backbone.forward_hidden(token_ids)
        return self.loss_head(hidden, labels)  # type: ignore[return-value]
