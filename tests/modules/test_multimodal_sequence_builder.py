"""Multimodal sequence fusion compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.multimodal_sequence_builder import MultimodalSequenceBuilder


def test_scatter_update_fusion_shape() -> None:
    seq, d, s_v = 128, 5120, 64
    graph = compose_graph(
        lambda _ctx: MultimodalSequenceBuilder(hidden_size=d, vocab_size=32000),
        (
            Tensor(shape=(seq,)),
            Tensor(shape=(s_v, d)),
            Tensor(shape=(s_v,)),
        ),
    )
    out_shape = graph.edge(graph.outputs[0]).tensor.shape
    assert out_shape == (seq, d)
    families = tuple(
        graph.node(node_id).operation_family for node_id in graph.nodes
    )
    assert "scatter_update" in families
    assert "embedding_lookup" in families
