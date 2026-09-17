"""MLP adapters from vision width to language hidden size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.compose import Module, Tensor

from .gelu import GELUErf, GELUTanh
from .linear import Linear
from .rms_norm import RMSNorm


@dataclass(frozen=True, slots=True)
class ProjectorConfig:
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    activation: Literal["gelu", "gelu_tanh"] = "gelu"
    final_rms_norm: bool = True
    scale_free_final_norm: bool = False


class MultimodalProjector(Module):
    """Depth-wise MLP projector with optional terminal RMSNorm."""

    module_kind = "MultimodalProjector"

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        sizes = (config.input_size, *config.hidden_sizes, config.output_size)
        if len(sizes) < 2:
            raise ValueError("projector requires at least input and output sizes")

        layers: list[Module] = []
        act = GELUErf() if config.activation == "gelu" else GELUTanh()
        for idx in range(len(sizes) - 1):
            layers.append(Linear(sizes[idx], sizes[idx + 1]))
            if idx < len(sizes) - 2:
                layers.append(act)
        if config.final_rms_norm:
            layers.append(
                RMSNorm(
                    config.output_size,
                    elementwise_affine=not config.scale_free_final_norm,
                )
            )
        self.layers = layers

    def forward(self, tokens: Tensor) -> Tensor:
        out = tokens
        for layer in self.layers:
            out = layer(out)  # type: ignore[assignment]
        return out  # type: ignore[return-value]


def muse_glimmer_perception_adapter() -> MultimodalProjector:
    return MultimodalProjector(
        ProjectorConfig(
            input_size=6144,
            hidden_sizes=(4096, 4096),
            output_size=6656,
            activation="gelu",
            scale_free_final_norm=True,
        )
    )


def gemma4_vision_projector() -> MultimodalProjector:
    return MultimodalProjector(
        ProjectorConfig(
            input_size=1152,
            hidden_sizes=(),
            output_size=5376,
            activation="gelu",
            final_rms_norm=False,
        )
    )


__all__ = [
    "MultimodalProjector",
    "ProjectorConfig",
    "gemma4_vision_projector",
    "muse_glimmer_perception_adapter",
]
