"""Multi-token prediction stage configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .layer_spec import LayerSpec
from .logit_soft_cap import LogitSoftCapConfig


@dataclass(frozen=True, slots=True)
class MtpStageConfig:
    """One auxiliary MTP stage (Qwen MLP head or Nemotron hybrid stack)."""

    hidden_size: int
    vocab_size: int
    num_prediction_steps: int
    source_hidden_layer_index: int
    layer_specs: tuple[LayerSpec, ...] = ()
    mlp_intermediate: int | None = None
    fusion_mode: Literal["concat_linear", "add"] = "concat_linear"
    tie_weights: bool = False
    soft_cap: LogitSoftCapConfig | None = None

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        if self.num_prediction_steps < 1:
            raise ValueError("num_prediction_steps must be >= 1")
        if self.source_hidden_layer_index < 0:
            raise ValueError("source_hidden_layer_index must be non-negative")
        if self.layer_specs:
            if self.mlp_intermediate is not None:
                raise ValueError("Nemotron path must not set mlp_intermediate")
        elif self.mlp_intermediate is None or self.mlp_intermediate <= 0:
            raise ValueError("Qwen path requires positive mlp_intermediate")


__all__ = ["MtpStageConfig"]
