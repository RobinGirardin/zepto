"""RoPEMaterialize cache shape and persistence tests."""

from __future__ import annotations

from zepto.compose import compose_graph
from modules.position.rope_config import RoPEConfig
from modules.position.rope_materialize import RoPEMaterialize


def _output_shapes(graph) -> tuple[tuple[int, ...], tuple[int, ...]]:
    outputs = graph.outputs
    cos_shape = graph.edge(outputs[0]).tensor.shape
    sin_shape = graph.edge(outputs[1]).tensor.shape
    return cos_shape, sin_shape


def test_default_full_cache_shapes() -> None:
    seq = 16
    head_dim = 128
    graph = compose_graph(lambda _ctx: RoPEMaterialize(seq, head_dim), ())
    cos_shape, sin_shape = _output_shapes(graph)
    assert cos_shape == (seq, head_dim)
    assert sin_shape == (seq, head_dim)


def test_partial_cache_shapes() -> None:
    seq = 16
    head_dim = 256
    rotary = 64
    graph = compose_graph(
        lambda _ctx: RoPEMaterialize(
            seq,
            head_dim,
            config=RoPEConfig(head_dim=head_dim, rotary_dim=rotary),
        ),
        (),
    )
    cos_shape, sin_shape = _output_shapes(graph)
    assert cos_shape == (seq, rotary)
    assert sin_shape == (seq, rotary)


def test_inv_freq_persistent() -> None:
    graph = compose_graph(lambda _ctx: RoPEMaterialize(8, 64), ())
    families = tuple(
        graph.node(node_id).operation_family for node_id in graph.nodes
    )
    assert "cos" in families
    assert "sin" in families
