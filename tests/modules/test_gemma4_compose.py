"""Gemma 4 full-model compose tests."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.gemma4 import Gemma4, Gemma4Config
from zepto.semantic.metadata import DType


def _tiny_gemma4(seq_len: int, *, include_vision: bool = True) -> Gemma4:
    return Gemma4(
        config=Gemma4Config(
            hidden_size=5376,
            swiglu_intermediate=21504,
            num_layers=2,
            vocab_size=512,
        ),
        seq_len=seq_len,
        include_vision=include_vision,
        vision_num_layers=1,
        grid_h=28,
        grid_w=40,
    )


def test_gemma4_text_only_skips_vision_compute() -> None:
    seq_len = 1024
    graph = compose_graph(
        lambda _ctx: _tiny_gemma4(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "Gemma4VisionPath" not in kinds
    assert "ScatterUpdate" not in kinds


def test_gemma4_text_only_still_counts_vision_parameters() -> None:
    seq_len = 1024
    graph = compose_graph(
        lambda _ctx: _tiny_gemma4(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    with_vision = estimate(
        graph, reference_invocation(default_dtype=DType.FP16)
    ).memory.breakdown.parameters
    graph_no = compose_graph(
        lambda _ctx: _tiny_gemma4(seq_len, include_vision=False),
        (Tensor(shape=(seq_len,)),),
    )
    without_vision = estimate(
        graph_no, reference_invocation(default_dtype=DType.FP16)
    ).memory.breakdown.parameters
    assert with_vision > without_vision


def test_gemma4_multimodal_includes_vision_path() -> None:
    seq_len = 1024
    graph = compose_graph(
        lambda _ctx: _tiny_gemma4(seq_len),
        (
            Tensor(shape=(seq_len,)),
            Tensor(shape=(280,), semantic_type="placeholder_indices"),
            Tensor(shape=(1, 28 * 16, 40 * 16, 3)),
        ),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "Gemma4VisionPath" in kinds
    text_only = compose_graph(
        lambda _ctx: _tiny_gemma4(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    ctx = reference_invocation(default_dtype=DType.FP16)
    assert (
        estimate(graph, ctx).flops.total_flops
        > estimate(text_only, ctx).flops.total_flops
    )
