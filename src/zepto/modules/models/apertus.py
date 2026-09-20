"""Apertus decoder-only model module."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.modules.blocks.apertus_decoder_block import ApertusDecoderBlock
from zepto.modules.layers.embedding import Embedding
from zepto.modules.layers.lm_head import LMHead
from zepto.modules.attention.materialized_causal_mask import MaterializedCausalMask
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_config import apertus_rope
from zepto.modules.position.rope_materialize import RoPEMaterialize


@dataclass(frozen=True, slots=True)
class ApertusConfig:
    """Architecture constants for preset factories."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int


APERTUS_8B = ApertusConfig(
    hidden_size=4096,
    intermediate_size=21504,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    vocab_size=131072,
)

APERTUS_70B = ApertusConfig(
    hidden_size=8192,
    intermediate_size=43008,
    num_heads=64,
    num_kv_heads=8,
    num_layers=80,
    vocab_size=131072,
)


class Apertus(Module):
    """Full Apertus decoder stack: embed → L blocks → final RMSNorm → LMHead."""

    module_kind = "Apertus"

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        vocab_size: int,
        seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")

        resolved_head_dim = (
            head_dim if head_dim is not None else hidden_size // num_heads
        )
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.embedding = Embedding(hidden_size, vocab_size)
        self.causal_mask = MaterializedCausalMask(seq_len)
        self.rope_materialize = RoPEMaterialize(
            seq_len,
            resolved_head_dim,
            config=apertus_rope(head_dim=resolved_head_dim),
        )

        self.blocks: list[ApertusDecoderBlock] = []
        for index in range(num_layers):
            block = ApertusDecoderBlock(
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim=resolved_head_dim,
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)

        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = LMHead(hidden_size, vocab_size)

    @classmethod
    def apertus_8b(cls, seq_len: int) -> Apertus:
        """Build the 8B preset; caller supplies sequence length."""
        cfg = APERTUS_8B
        return cls(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            seq_len=seq_len,
        )

    @classmethod
    def apertus_70b(cls, seq_len: int) -> Apertus:
        """Build the 70B preset; caller supplies sequence length."""
        cfg = APERTUS_70B
        return cls(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            seq_len=seq_len,
        )

    def forward_hidden(self, token_ids: Tensor) -> Tensor:
        """Backbone hidden states before LM head (embed → blocks → final_norm)."""
        hidden_states = self.embedding(token_ids)
        causal_mask = self.causal_mask()
        rope_cos, rope_sin = self.rope_materialize()

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                causal_mask,
                rope_cos,
                rope_sin,
            )  # type: ignore[assignment]

        return self.final_norm(hidden_states)  # type: ignore[return-value]

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.lm_head(self.forward_hidden(token_ids))  # type: ignore[return-value]
