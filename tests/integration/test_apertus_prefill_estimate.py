"""Phase 3 integration: Apertus prefill cost estimation."""

from __future__ import annotations

from zepto.analysis import account_memory, estimate, lower, reference_invocation
from zepto.analysis.horizon.state import StatePortRegistry
from zepto.compose import Tensor, compose_graph
from modules.models.apertus import Apertus
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
