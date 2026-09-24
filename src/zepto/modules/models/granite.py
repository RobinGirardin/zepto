"""IBM Granite 4.2 decoder-only full model."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from zepto.modules._internal._batch import expect_token_ids_rank
from zepto.modules.attention.attention_config import AttentionConfig, granite_attention_layer
from zepto.modules.attention.decoder_attention_context import DecoderAttentionContext
from zepto.modules.layers.embedding import Embedding
from zepto.modules.output.output_presets import granite_lm_output
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_config import bind_rope, granite_rope
from zepto.modules.blocks.swiglu_decoder_block import SwiGLUDecoderBlock


@dataclass(frozen=True, slots=True)
class GraniteConfig:
    """Architecture constants for Granite 4.2 30B."""

    hidden_size: int = 4096
    intermediate_size: int = 32768
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 64
    vocab_size: int = 100352


GRANITE_42_30B = GraniteConfig()


class Granite(Module):
    """Embed → SwiGLU blocks → final RMSNorm → untied LM head."""

    module_kind = "Granite"

    def __init__(
        self,
        *,
        config: GraniteConfig = GRANITE_42_30B,
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
        attn = (
            granite_attention_layer()
            if cfg == GRANITE_42_30B
            else AttentionConfig(
                hidden_size=cfg.hidden_size,
                num_q_heads=cfg.num_q_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
            )
        )
        rope_apply = RoPEApply(cfg.head_dim)
        self.attn_context = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda _i: bind_rope(
                attn, granite_rope(head_dim=cfg.head_dim)
            ),
            num_layers=layers,
        )
        self.blocks: list[SwiGLUDecoderBlock] = []
        for index in range(layers):
            block = SwiGLUDecoderBlock(
                attn,
                swiglu_intermediate=cfg.intermediate_size,
                rope=rope_apply,
                qk_norm=None,
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = granite_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )

    @classmethod
    def granite_42_30b(cls, seq_len: int) -> Granite:
        return cls(config=GRANITE_42_30B, seq_len=seq_len)

    def forward_hidden(self, token_ids: Tensor) -> Tensor:
        """Backbone hidden states before LM head (embed → blocks → final_norm)."""
        expect_token_ids_rank(token_ids)
        hidden = self.embedding(token_ids)
        for index, block in enumerate(self.blocks):
            ctx = self.attn_context.for_layer(index)
            hidden = block(
                hidden,
                ctx.mask,
                ctx.cos,
                ctx.sin,
            )  # type: ignore[assignment]
        return self.final_norm(hidden)  # type: ignore[return-value]

    def forward(self, token_ids: Tensor) -> Tensor:
        """Compose on rank-1 ``(S,)`` or rank-2 ``(B, S)`` token ids."""
        return self.lm_output(self.forward_hidden(token_ids))  # type: ignore[return-value]


__all__ = ["GRANITE_42_30B", "Granite", "GraniteConfig"]
