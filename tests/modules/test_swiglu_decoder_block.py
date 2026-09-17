"""SwiGLUDecoderBlock compose smoke tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from modules.attention.attention_config import granite_attention_layer
from modules.attention.materialized_causal_mask import MaterializedCausalMask
from modules.position.rope_apply import RoPEApply
from modules.position.rope_materialize import RoPEMaterialize
from modules.blocks.swiglu_decoder_block import SwiGLUDecoderBlock


def test_swiglu_decoder_block_output_rank_matches_input() -> None:
    seq_len, hidden = 32, 128
    attn = granite_attention_layer()
    attn = attn.__class__(
        hidden_size=hidden,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
    )

    def factory(_ctx):
        block = SwiGLUDecoderBlock(
            attn,
            swiglu_intermediate=256,
            rope=RoPEApply(32),
        )
        mask_mod = MaterializedCausalMask(seq_len)
        rope_mod = RoPEMaterialize(seq_len, 32)

        class _Harness(Module):
            module_kind = "SwiGLUHarness"

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
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (seq_len, hidden)
