"""Qwen3.8 text-only training model with fused linear cross-entropy loss."""

from __future__ import annotations

from zepto.compose import Module, Tensor

from .models.qwen38 import QWEN38_27B, Qwen38, Qwen38Config
from .output.fused_linear_cross_entropy import FusedLinearCrossEntropy
from .position.rope_config import RoPEConfig


class Qwen38ForCausalLM(Module):
    """Training wrapper: text-only backbone hidden + shared-weight fused CE.

    Vision and MTP are hard-off. This is the scored study / horizon graph,
    not the production VL+MTP module.
    """

    module_kind = "Qwen38ForCausalLM"

    def __init__(
        self,
        *,
        config: Qwen38Config = QWEN38_27B,
        seq_len: int,
        num_layers: int | None = None,
        layer_specs: tuple | None = None,
        rope_config: RoPEConfig | None = None,
        include_vision: bool = False,
        include_mtp: bool = False,
    ) -> None:
        super().__init__()
        if include_vision or include_mtp:
            raise ValueError(
                "Qwen38ForCausalLM is text-only (include_vision and include_mtp must be False)"
            )
        self.backbone = Qwen38(
            config=config,
            seq_len=seq_len,
            num_layers=num_layers,
            include_vision=False,
            include_mtp=False,
            layer_specs=layer_specs,
            rope_config=rope_config,
        )
        self.loss_head = FusedLinearCrossEntropy(
            config.hidden_size,
            config.vocab_size,
            weight=self.backbone.lm_output.weight,
        )

    def forward(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        hidden = self.backbone.forward_hidden(token_ids)
        return self.loss_head(hidden, labels)  # type: ignore[return-value]
