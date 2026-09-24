"""Apertus 1.5 text-only decoder module (wider embed, pruned LM head)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.modules._internal._batch import expect_token_ids_rank
from zepto.modules.blocks.apertus_decoder_block import ApertusDecoderBlock
from zepto.modules.layers.embedding import Embedding
from zepto.modules.layers.lm_head import LMHead
from zepto.modules.attention.materialized_causal_mask import MaterializedCausalMask
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_config import apertus_rope
from zepto.modules.position.rope_materialize import RoPEMaterialize


@dataclass(frozen=True, slots=True)
class Apertus15Config:
    """Architecture constants for Apertus 1.5 text-only preset factories."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    output_vocab_size: int


APERTUS15_8B = Apertus15Config(
    hidden_size=4096,
    intermediate_size=21504,
    num_heads=32,
    num_kv_heads=8,
    num_layers=32,
    vocab_size=266752,
    output_vocab_size=131072,
)

APERTUS15_70B = Apertus15Config(
    hidden_size=8192,
    intermediate_size=43008,
    num_heads=64,
    num_kv_heads=8,
    num_layers=80,
    vocab_size=266752,
    output_vocab_size=131072,
)


class Apertus15(Module):
    """Text-only Apertus 1.5 decoder: embed → L blocks → final RMSNorm → LMHead.

    Same decoder as ``Apertus`` (1.0). The costing delta is a vocab split:
    input ``Embedding`` rows are ``vocab_size``; ``LMHead`` columns are
    ``output_vocab_size``. Embeddings stay untied.

    Architecture-only constructor: ``seq_len`` sizes the causal mask and
    RoPE caches. Parallel batch is never a constructor argument; pass
    ``(B, S)`` (or legacy ``(S,)``) token ids to :meth:`forward`.

    ::

        Apertus15(
            *,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            num_kv_heads: int,
            num_layers: int,
            vocab_size: int,            # input Embedding rows
            output_vocab_size: int,     # LMHead columns
            seq_len: int,
            head_dim: int | None = None,
        ) -> Module

        Apertus15.forward(token_ids) -> logits   # (S, V_out) or (B, S, V_out)
        Apertus15.forward_hidden(token_ids) -> hidden  # (S, H) or (B, S, H)
    """

    module_kind = "Apertus15"

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        vocab_size: int,
        output_vocab_size: int,
        seq_len: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        if vocab_size <= 0 or output_vocab_size <= 0:
            raise ValueError("vocab_size and output_vocab_size must be positive")
        if output_vocab_size > vocab_size:
            raise ValueError(
                "output head cannot exceed input embed "
                f"(output_vocab_size={output_vocab_size} > vocab_size={vocab_size})"
            )

        resolved_head_dim = (
            head_dim if head_dim is not None else hidden_size // num_heads
        )
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size

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
        self.lm_head = LMHead(hidden_size, output_vocab_size)

    @classmethod
    def apertus15_8b(cls, seq_len: int) -> Apertus15:
        """Build the 8B text-only preset; caller supplies sequence length."""
        cfg = APERTUS15_8B
        return cls(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            output_vocab_size=cfg.output_vocab_size,
            seq_len=seq_len,
        )

    @classmethod
    def apertus15_70b(cls, seq_len: int) -> Apertus15:
        """Build the 70B text-only preset; caller supplies sequence length."""
        cfg = APERTUS15_70B
        return cls(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            output_vocab_size=cfg.output_vocab_size,
            seq_len=seq_len,
        )

    def forward_hidden(self, token_ids: Tensor) -> Tensor:
        """Backbone hidden states before the pruned LM head."""
        expect_token_ids_rank(token_ids)
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
        """Run the decoder on ``(S,)`` or batched ``(B, S)`` token ids."""
        return self.lm_head(self.forward_hidden(token_ids))  # type: ignore[return-value]


__all__ = ["APERTUS15_70B", "APERTUS15_8B", "Apertus15", "Apertus15Config"]
