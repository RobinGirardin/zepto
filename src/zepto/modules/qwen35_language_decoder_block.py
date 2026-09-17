"""Qwen3.8 language decoder block: mixer + SwiGLU."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .gated_delta_net import GatedDeltaNet
from .layer_spec import LayerSpec
from .rms_norm import RMSNorm
from .swiglu import SwiGLU


class Qwen35LanguageDecoderBlock(Module):
    """Pre-norm mixer residual, then pre-norm SwiGLU residual."""

    module_kind = "Qwen35LanguageDecoderBlock"

    def __init__(self, spec: LayerSpec) -> None:
        super().__init__()
        if spec.mixer not in ("gated_delta_net", "attention"):
            raise ValueError("Qwen language block supports delta or attention mixers")
        if spec.ffn != "swiglu" or spec.swiglu_intermediate is None:
            raise ValueError("Qwen language block requires SwiGLU FFN")
        hidden = (
            spec.gated_delta.hidden_size
            if spec.gated_delta is not None
            else spec.attention.hidden_size  # type: ignore[union-attr]
        )
        self.pre_mixer_norm = RMSNorm(hidden)
        self.pre_ffn_norm = RMSNorm(hidden)
        if spec.mixer == "gated_delta_net":
            assert spec.gated_delta is not None
            self.mixer = GatedDeltaNet(spec.gated_delta)
        else:
            assert spec.attention is not None
            self.mixer = FlexibleAttention(spec.attention)
        self.ffn = SwiGLU(hidden, spec.swiglu_intermediate)
        self._mixer_kind = spec.mixer

    def forward(
        self,
        hidden_states: Tensor,
        *mixer_inputs: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normed = self.pre_mixer_norm(hidden_states)
        if self._mixer_kind == "gated_delta_net":
            conv_in = mixer_inputs[0] if len(mixer_inputs) > 0 else None
            scan_in = mixer_inputs[1] if len(mixer_inputs) > 1 else None
            mixer_out, conv_out, scan_out = self.mixer(normed, conv_in, scan_in)  # type: ignore[misc]
            hidden = Add()(residual, mixer_out)  # type: ignore[assignment]
            ffn_out = self.ffn(self.pre_ffn_norm(hidden))
            out = Add()(hidden, ffn_out)  # type: ignore[assignment]
            return out, conv_out, scan_out
        mask = mixer_inputs[0]
        cos = mixer_inputs[1] if len(mixer_inputs) > 1 else None
        sin = mixer_inputs[2] if len(mixer_inputs) > 2 else None
        mixer_out = self.mixer(normed, mask, cos, sin)  # type: ignore[misc]
        hidden = Add()(residual, mixer_out)  # type: ignore[assignment]
        ffn_out = self.ffn(self.pre_ffn_norm(hidden))
        return Add()(hidden, ffn_out)  # type: ignore[return-value]


__all__ = ["Qwen35LanguageDecoderBlock"]
