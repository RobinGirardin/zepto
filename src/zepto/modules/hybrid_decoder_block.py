"""Single-mixer decoder block (Nemotron-style)."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .layer_spec import LayerSpec
from .mamba2_mixer import Mamba2Mixer
from .gated_delta_net import GatedDeltaNet
from .rms_norm import RMSNorm


class HybridDecoderBlock(Module):
    """RMSNorm → one mixer → residual."""

    module_kind = "HybridDecoderBlock"

    def __init__(
        self,
        spec: LayerSpec,
        *,
        moe_factory: Callable[..., Module] | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden = _hidden_size(spec)
        self.pre_norm = RMSNorm(hidden)
        if spec.mixer == "mamba2":
            assert spec.mamba2 is not None
            self.mixer = Mamba2Mixer(spec.mamba2)
        elif spec.mixer == "gated_delta_net":
            assert spec.gated_delta is not None
            self.mixer = GatedDeltaNet(spec.gated_delta)
        elif spec.mixer == "attention":
            assert spec.attention is not None
            self.mixer = FlexibleAttention(spec.attention)
        elif spec.mixer == "moe":
            if moe_factory is None:
                raise ValueError("moe mixer requires moe_factory")
            self.mixer = moe_factory()
        else:
            raise ValueError(f"unsupported mixer {spec.mixer}")

    def forward(
        self,
        hidden_states: Tensor,
        *mixer_inputs: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normed = self.pre_norm(hidden_states)
        spec = self.spec
        if spec.mixer in ("mamba2", "gated_delta_net"):
            conv_in = mixer_inputs[0] if len(mixer_inputs) > 0 else None
            scan_in = mixer_inputs[1] if len(mixer_inputs) > 1 else None
            mixer_out, conv_out, scan_out = self.mixer(normed, conv_in, scan_in)  # type: ignore[misc]
            out = Add()(residual, mixer_out)  # type: ignore[assignment]
            return out, conv_out, scan_out
        if spec.mixer == "attention":
            mask = mixer_inputs[0]
            cos = mixer_inputs[1] if len(mixer_inputs) > 1 else None
            sin = mixer_inputs[2] if len(mixer_inputs) > 2 else None
            mixer_out = self.mixer(normed, mask, cos, sin)  # type: ignore[misc]
            return Add()(residual, mixer_out)  # type: ignore[return-value]
        mixer_out = self.mixer(normed)  # type: ignore[misc]
        return Add()(residual, mixer_out)  # type: ignore[return-value]


def _hidden_size(spec: LayerSpec) -> int:
    if spec.gated_delta is not None:
        return spec.gated_delta.hidden_size
    if spec.mamba2 is not None:
        return spec.mamba2.hidden_size
    if spec.attention is not None:
        return spec.attention.hidden_size
    if spec.moe_hidden_size is not None:
        return spec.moe_hidden_size
    raise ValueError("cannot infer hidden size from layer spec")


__all__ = ["HybridDecoderBlock"]
