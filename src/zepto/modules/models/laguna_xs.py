"""Laguna XS 2.1 decoder-only full model."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from zepto.modules.attention.decoder_attention_context import DecoderAttentionContext
from zepto.modules.layers.embedding import Embedding
from zepto.modules.blocks.laguna_decoder_block import LagunaDecoderBlock
from zepto.modules.moe.moe_presets import laguna_sparse_moe_block
from zepto.modules.output.output_presets import laguna_xs_lm_output
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_config import laguna_layer_binding


@dataclass(frozen=True, slots=True)
class LagunaXsConfig:
    """Architecture constants for Laguna XS 2.1."""

    hidden_size: int = 2048
    dense_swiglu_intermediate: int = 8192
    num_layers: int = 40
    vocab_size: int = 100352


LAGUNA_XS_21 = LagunaXsConfig()


def _rope_apply_for_binding(layer_index: int) -> RoPEApply | None:
    binding = laguna_layer_binding(layer_index=layer_index)
    if binding.rope is None:
        return None
    cfg = binding.rope
    rotary = cfg.resolved_rotary_dim
    if rotary == cfg.head_dim:
        return RoPEApply(cfg.head_dim)
    return RoPEApply(cfg.head_dim, rotary_dim=rotary)


class LagunaXs(Module):
    """Embed → Laguna blocks → final RMSNorm → untied LM head."""

    module_kind = "LagunaXs"

    def __init__(
        self,
        *,
        config: LagunaXsConfig = LAGUNA_XS_21,
        seq_len: int,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        cfg = config
        layers = num_layers if num_layers is not None else cfg.num_layers
        if layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        self.seq_len = seq_len
        self.config = cfg
        self.embedding = Embedding(cfg.hidden_size, cfg.vocab_size)
        self.attn_context = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda i: laguna_layer_binding(layer_index=i),
            num_layers=layers,
        )
        self.blocks: list[LagunaDecoderBlock] = []
        for index in range(layers):
            binding = laguna_layer_binding(layer_index=index)
            block = LagunaDecoderBlock(
                index,
                attention=binding.attention,
                hidden_size=cfg.hidden_size,
                rope=_rope_apply_for_binding(index),
                dense_swiglu_intermediate=cfg.dense_swiglu_intermediate,
                moe_factory=(
                    None
                    if index == 0
                    else (lambda s=seq_len: laguna_sparse_moe_block(seq_len=s))
                ),
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = laguna_xs_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )

    @classmethod
    def laguna_xs_21(cls, seq_len: int) -> LagunaXs:
        return cls(config=LAGUNA_XS_21, seq_len=seq_len)

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embedding(token_ids)
        for index, block in enumerate(self.blocks):
            ctx = self.attn_context.for_layer(index)
            hidden = block(
                hidden,
                ctx.mask,
                ctx.cos,
                ctx.sin,
            )  # type: ignore[assignment]
        return self.lm_output(self.final_norm(hidden))  # type: ignore[return-value]


__all__ = ["LAGUNA_XS_21", "LagunaXs", "LagunaXsConfig"]
