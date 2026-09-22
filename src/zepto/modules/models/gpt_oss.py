"""GPT-OSS 20B decoder-only full model."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from zepto.modules._internal._batch import expect_token_ids_rank
from zepto.modules.blocks.attention_moe_decoder_block import AttentionMoEDecoderBlock
from zepto.modules.attention.decoder_attention_context import DecoderAttentionContext
from zepto.modules.layers.embedding import Embedding
from zepto.modules.moe.moe_presets import gpt_oss_moe_block
from zepto.modules.output.output_presets import gpt_oss_lm_output
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_config import gpt_oss_layer_binding


@dataclass(frozen=True, slots=True)
class GptOssConfig:
    """Architecture constants for GPT-OSS 20B."""

    hidden_size: int = 2880
    num_layers: int = 24
    vocab_size: int = 201088


GPT_OSS_20B = GptOssConfig()


class GptOss(Module):
    """Embed → attention+MoE blocks → final RMSNorm → untied LM head."""

    module_kind = "GptOss"

    def __init__(
        self,
        *,
        config: GptOssConfig = GPT_OSS_20B,
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
            layer_binding=lambda i: gpt_oss_layer_binding(layer_index=i),
            num_layers=layers,
        )
        self.blocks: list[AttentionMoEDecoderBlock] = []
        for index in range(layers):
            block = AttentionMoEDecoderBlock(
                index,
                seq_len=seq_len,
                moe_factory=lambda: gpt_oss_moe_block(seq_len=seq_len),
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = gpt_oss_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )

    @classmethod
    def gpt_oss_20b(cls, seq_len: int) -> GptOss:
        return cls(config=GPT_OSS_20B, seq_len=seq_len)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Compose on rank-1 ``(S,)`` or rank-2 ``(B, S)`` token ids."""
        expect_token_ids_rank(token_ids)
        hidden = self.embedding(token_ids)
        for index, block in enumerate(self.blocks):
            ctx = self.attn_context.for_layer(index)
            hidden = block(
                hidden,
                ctx.mask,
                ctx.cos,  # type: ignore[arg-type]
                ctx.sin,  # type: ignore[arg-type]
            )  # type: ignore[assignment]
        return self.lm_output(self.final_norm(hidden))  # type: ignore[return-value]


__all__ = ["GPT_OSS_20B", "GptOss", "GptOssConfig"]
