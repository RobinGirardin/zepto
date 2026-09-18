"""MultimodalRoPEMaterialize graph shape tests."""

from __future__ import annotations

import pytest

from zepto.compose import compose_graph
from modules.position.multimodal_rope_materialize import MultimodalRoPEMaterialize
from modules.position.rope_config import RoPEConfig, qwen3_vl_mrope


def test_mrope_output_shapes() -> None:
    seq = 8
    cfg = qwen3_vl_mrope()
    graph = compose_graph(
        lambda _ctx: MultimodalRoPEMaterialize(seq, cfg),
        (),
    )
    cos_shape = graph.edge(graph.outputs[0]).tensor.shape
    sin_shape = graph.edge(graph.outputs[1]).tensor.shape
    assert cos_shape == (seq, 64)
    assert sin_shape == (seq, 64)


def test_wrong_rope_type_raises() -> None:
    cfg = RoPEConfig(head_dim=256, rope_type="default")
    with pytest.raises(ValueError, match="mrope"):
        MultimodalRoPEMaterialize(8, cfg)
