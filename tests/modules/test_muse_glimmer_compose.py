"""Muse Glimmer full-model compose tests."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from modules.models.muse_glimmer import MuseGlimmer, MuseGlimmerConfig
from zepto.semantic.metadata import DType


def _tiny_muse(seq_len: int) -> MuseGlimmer:
    return MuseGlimmer(
        config=MuseGlimmerConfig(
            hidden_size=6656,
            swiglu_intermediate=19968,
            num_layers=2,
            vocab_size=512,
        ),
        seq_len=seq_len,
        vision_num_layers=1,
    )


def test_muse_glimmer_text_only_no_vision_compute() -> None:
    seq_len = 2048
    graph = compose_graph(
        lambda _ctx: _tiny_muse(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "MuseGlimmerVisionTower" not in kinds


def test_muse_glimmer_multimodal_flops_exceed_text_only() -> None:
    seq_len = 2048
    text = compose_graph(
        lambda _ctx: _tiny_muse(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    mm = compose_graph(
        lambda _ctx: _tiny_muse(seq_len),
        (
            Tensor(shape=(seq_len,)),
            Tensor(shape=(49,), semantic_type="placeholder_indices"),
            Tensor(shape=(2, 14 * 14, 14 * 14, 3)),
        ),
    )
    ctx = reference_invocation(default_dtype=DType.FP16)
    assert estimate(mm, ctx).flops.total_flops > estimate(text, ctx).flops.total_flops
