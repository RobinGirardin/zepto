"""SandwichNormSwiGLUDecoderBlock compose smoke tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from modules.attention.attention_config import granite_attention_layer
from modules.attention.materialized_causal_mask import MaterializedCausalMask
from modules.position.rope_apply import RoPEApply
from modules.position.rope_materialize import RoPEMaterialize
from modules.blocks.sandwich_norm_swiglu_decoder_block import (
    SandwichNormSwiGLUDecoderBlock,
)


def test_sandwich_block_has_four_rmsnorms_per_layer() -> None:
    seq_len, hidden = 8, 256
    head_dim = 64
    attn = granite_attention_layer()
    attn = attn.__class__(
        hidden_size=hidden,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
    )

    def factory(_ctx):
        block = SandwichNormSwiGLUDecoderBlock(
            attn,
            swiglu_intermediate=512,
            rope=RoPEApply(head_dim),
        )
        mask_mod = MaterializedCausalMask(seq_len)
        rope_mod = RoPEMaterialize(seq_len, head_dim)

        class _Harness(Module):
            module_kind = "SandwichHarness"

            def __init__(self) -> None:
                super().__init__()
                self.block = block
                self.mask_mod = mask_mod
                self.rope_mod = rope_mod

            def forward(self, hidden_states: Tensor) -> Tensor:
                cos, sin = self.rope_mod()
                return self.block(hidden_states, self.mask_mod(), cos, sin)

        return _Harness()

    graph = compose_graph(
        factory,
        (Tensor(shape=(seq_len, hidden), requires_grad=True),),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert kinds.count("RMSNorm") >= 4
