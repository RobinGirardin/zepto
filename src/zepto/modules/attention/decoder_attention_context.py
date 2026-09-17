"""Per-layer causal masks and RoPE caches for full decoder models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zepto.compose import Module, Tensor

from .materialized_causal_mask import MaterializedCausalMask
from zepto.modules.position.rope_config import LayerRoPEBinding
from zepto.modules.position.rope_materialize import RoPEMaterialize
from .sliding_window_causal_mask import SlidingWindowCausalMask


@dataclass(frozen=True, slots=True)
class LayerAttentionInputs:
    """Mask and RoPE tensors for one decoder layer."""

    binding: LayerRoPEBinding
    mask: Tensor
    cos: Tensor | None
    sin: Tensor | None


class DecoderAttentionContext(Module):
    """Pre-register per-layer masks and RoPE caches for one ``seq_len``."""

    module_kind = "DecoderAttentionContext"

    def __init__(
        self,
        seq_len: int,
        *,
        layer_binding: Callable[[int], LayerRoPEBinding],
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.seq_len = seq_len
        self._binding_fn = layer_binding
        self._masks: list[Module] = []
        self._ropes: list[RoPEMaterialize | None] = []
        for index in range(num_layers):
            binding = layer_binding(index)
            attn = binding.attention
            if attn.mask == "sliding":
                mask_mod = SlidingWindowCausalMask(
                    seq_len, attn.window_size  # type: ignore[arg-type]
                )
            else:
                mask_mod = MaterializedCausalMask(seq_len)
            self.register_module(f"masks.{index}", mask_mod)
            self._masks.append(mask_mod)
            if binding.rope is None:
                self._ropes.append(None)
            else:
                cfg = binding.rope
                rope_mod = RoPEMaterialize(seq_len, cfg.head_dim, config=cfg)
                self.register_module(f"ropes.{index}", rope_mod)
                self._ropes.append(rope_mod)

    def for_layer(self, layer_index: int) -> LayerAttentionInputs:
        binding = self._binding_fn(layer_index)
        mask_mod = self._masks[layer_index]
        rope_mod = self._ropes[layer_index]
        mask = mask_mod()  # type: ignore[misc]
        if rope_mod is None:
            return LayerAttentionInputs(binding, mask, None, None)
        cos, sin = rope_mod()
        return LayerAttentionInputs(binding, mask, cos, sin)


__all__ = ["DecoderAttentionContext", "LayerAttentionInputs"]
