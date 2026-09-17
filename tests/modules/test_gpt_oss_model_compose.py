"""GPT-OSS full-model compose tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.decoder_attention_context import DecoderAttentionContext
from zepto.modules.gpt_oss import GptOss, GptOssConfig
from zepto.modules.rope_config import gpt_oss_layer_binding


def _tiny_gpt_oss(seq_len: int) -> GptOss:
    return GptOss(
        config=GptOssConfig(hidden_size=2880, num_layers=2, vocab_size=512),
        seq_len=seq_len,
    )


def _mask_kinds(layer_index: int, seq_len: int) -> set[str]:
    def factory(_ctx):
        ctx_mod = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda i: gpt_oss_layer_binding(layer_index=i),
            num_layers=2,
        )

        class _Probe(Module):
            module_kind = "MaskProbe"

            def __init__(self) -> None:
                super().__init__()
                self.ctx_mod = ctx_mod
                self.layer_index = layer_index

            def forward(self) -> Tensor:
                return self.ctx_mod.for_layer(self.layer_index).mask

        return _Probe()

    graph = compose_graph(factory, ())
    return {graph.node(n).provenance.component_type for n in graph.nodes}


def test_gpt_oss_even_layer_sliding_mask() -> None:
    assert "SlidingWindowCausalMask" in _mask_kinds(0, 128)


def test_gpt_oss_odd_layer_full_mask() -> None:
    assert "MaterializedCausalMask" in _mask_kinds(1, 128)


def test_gpt_oss_tiny_compose() -> None:
    seq_len = 128
    graph = compose_graph(
        lambda _ctx: _tiny_gpt_oss(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    assert graph.edge(graph.outputs[0]).tensor.shape == (seq_len, 512)
