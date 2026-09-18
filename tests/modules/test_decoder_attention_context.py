"""DecoderAttentionContext mask binding smoke tests."""

from __future__ import annotations

from zepto.compose import Module, Tensor, compose_graph
from modules.attention.decoder_attention_context import DecoderAttentionContext
from modules.position.rope_config import gpt_oss_layer_binding


def _mask_component_types(graph) -> set[str]:
    return {graph.node(node_id).provenance.component_type for node_id in graph.nodes}


def test_gpt_oss_layer0_uses_sliding_window_mask() -> None:
    seq_len = 128

    def factory(_ctx):
        ctx_mod = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda i: gpt_oss_layer_binding(layer_index=i),
            num_layers=2,
        )

        class _Probe(Module):
            module_kind = "MaskProbe0"

            def __init__(self) -> None:
                super().__init__()
                self.ctx_mod = ctx_mod

            def forward(self) -> Tensor:
                return self.ctx_mod.for_layer(0).mask

        return _Probe()

    graph = compose_graph(factory, ())
    assert "SlidingWindowCausalMask" in _mask_component_types(graph)


def test_gpt_oss_layer1_uses_full_causal_mask() -> None:
    seq_len = 128

    def factory(_ctx):
        ctx_mod = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda i: gpt_oss_layer_binding(layer_index=i),
            num_layers=2,
        )

        class _Probe(Module):
            module_kind = "MaskProbe1"

            def __init__(self) -> None:
                super().__init__()
                self.ctx_mod = ctx_mod

            def forward(self) -> Tensor:
                return self.ctx_mod.for_layer(1).mask

        return _Probe()

    graph = compose_graph(factory, ())
    assert "MaterializedCausalMask" in _mask_component_types(graph)
