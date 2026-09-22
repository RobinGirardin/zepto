"""Phase 3 integration: Apertus prefill cost estimation."""

from __future__ import annotations

import pytest

from zepto.analysis import account_memory, estimate, lower, reference_invocation
from zepto.analysis.horizon.state import StatePortRegistry
from zepto.compose import Tensor, compose_graph
from zepto.modules.models.apertus import Apertus
from zepto.semantic.metadata import DType


def _tiny_apertus(seq_len: int) -> Apertus:
    return Apertus(
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        vocab_size=1024,
        seq_len=seq_len,
    )


def _flash_ctx(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
        "default_dtype": DType.FP16,
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def test_apertus_prefill_estimate_produces_cost_report() -> None:
    seq_len = 64
    graph = compose_graph(
        lambda _ctx: _tiny_apertus(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    report = estimate(graph, _flash_ctx())

    assert report.flops.total_flops > 0
    assert report.memory.peak_live_bytes > 0
    assert report.memory.breakdown.parameters > 0

    _, lowered = estimate(graph, _flash_ctx(), return_lowered=True)
    gqa_regions = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa")
    ]
    assert len(gqa_regions) == 2


def test_apertus_prefill_with_kv_template_emits_state_port_events() -> None:
    seq_len = 32
    graph = compose_graph(
        lambda _ctx: _tiny_apertus(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    registry = StatePortRegistry.empty().configure_kv(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        dtype=DType.FP16,
    )
    lowered = lower(graph, _flash_ctx(), state_ports=registry)

    assert len(lowered.state_port_events) == 2
    assert all(event.bytes > 0 for event in lowered.state_port_events)

    mem = account_memory(lowered)
    assert mem.breakdown.state > 0


def test_apertus_prefill_estimate_batched_peak_scales_with_batch() -> None:
    B = 2
    seq_len = 64
    tokens = dict(semantic_type="token_ids", requires_grad=False)
    ctx = _flash_ctx()
    graph_b = compose_graph(
        lambda _ctx: _tiny_apertus(seq_len),
        (Tensor(shape=(B, seq_len), **tokens),),
    )
    graph_1 = compose_graph(
        lambda _ctx: _tiny_apertus(seq_len),
        (Tensor(shape=(1, seq_len), **tokens),),
    )
    peak_b = estimate(graph_b, ctx).memory.peak_live_bytes
    peak_1 = estimate(graph_1, ctx).memory.peak_live_bytes
    params = estimate(graph_1, ctx).memory.breakdown.parameters
    # Params do not scale with B. Shared mask is (1, S, S); ratio the
    # transient/activation remainder, not raw peak. See ADR-0010.
    assert (peak_b - params) / (peak_1 - params) == pytest.approx(B, rel=0.05)
    assert peak_b > peak_1

    flops_b = estimate(graph_b, ctx).flops.total_flops
    flops_1 = estimate(graph_1, ctx).flops.total_flops
    assert flops_b / flops_1 == pytest.approx(B, rel=0.05)


def test_apertus_prefill_batched_kv_state_port_events() -> None:
    B = 2
    seq_len = 32
    tokens = dict(semantic_type="token_ids", requires_grad=False)
    graph = compose_graph(
        lambda _ctx: _tiny_apertus(seq_len),
        (Tensor(shape=(B, seq_len), **tokens),),
    )
    registry = StatePortRegistry.empty().configure_kv(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        dtype=DType.FP16,
    )
    lowered = lower(graph, _flash_ctx(), state_ports=registry)

    itemsize = DType.FP16.itemsize or 0
    expected_per_layer = 2 * B * 2 * seq_len * 32 * itemsize
    assert len(lowered.state_port_events) == 2
    assert all(event.bytes == expected_per_layer for event in lowered.state_port_events)
