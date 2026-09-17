"""Per-layer mixer schedule specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.modules.attention.attention_config import AttentionConfig
from .mixer_config import GatedDeltaNetConfig, Mamba2MixerConfig

MixerKind = Literal["gated_delta_net", "mamba2", "attention", "moe"]
FfnKind = Literal["swiglu", "none"]


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """Select one mixer and optional FFN for a decoder layer."""

    mixer: MixerKind
    gated_delta: GatedDeltaNetConfig | None = None
    mamba2: Mamba2MixerConfig | None = None
    attention: AttentionConfig | None = None
    ffn: FfnKind = "none"
    swiglu_intermediate: int | None = None
    moe_hidden_size: int | None = None

    def __post_init__(self) -> None:
        if self.mixer == "gated_delta_net":
            if self.gated_delta is None:
                raise ValueError("gated_delta_net requires gated_delta config")
            if self.mamba2 is not None or self.attention is not None:
                raise ValueError("unexpected extra mixer config")
        elif self.mixer == "mamba2":
            if self.mamba2 is None:
                raise ValueError("mamba2 requires mamba2 config")
            if self.gated_delta is not None or self.attention is not None:
                raise ValueError("unexpected extra mixer config")
        elif self.mixer == "attention":
            if self.attention is None:
                raise ValueError("attention requires attention config")
            if self.gated_delta is not None or self.mamba2 is not None:
                raise ValueError("unexpected extra mixer config")
        elif self.mixer == "moe":
            if self.gated_delta is not None or self.mamba2 is not None or self.attention is not None:
                raise ValueError("moe must not carry attention/mixer configs")
            if self.moe_hidden_size is None:
                raise ValueError("moe requires moe_hidden_size")
        else:
            raise ValueError(f"unknown mixer kind {self.mixer!r}")

        if self.ffn == "swiglu":
            if self.swiglu_intermediate is None:
                raise ValueError("swiglu ffn requires swiglu_intermediate")
        elif self.swiglu_intermediate is not None:
            raise ValueError("swiglu_intermediate set without swiglu ffn")


__all__ = ["FfnKind", "LayerSpec", "MixerKind"]
