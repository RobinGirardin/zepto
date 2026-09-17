"""Discovery, lowering, and variant tests for region/gated_rms_norm."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.gated_delta_net import GatedDeltaNet
from zepto.modules.gated_rms_norm import GatedRMSNorm
from zepto.modules.mixer_config import GatedDeltaNetConfig
from zepto.semantic import ResourceEventKind


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def _gated_rms_graph(*, shape: tuple[int, int] = (4, 8)):
    return compose_graph(
        lambda ctx: GatedRMSNorm(shape[1]),
        (
            Tensor(shape=shape, requires_grad=True),
            Tensor(shape=shape, requires_grad=True),
        ),
    )


def test_compose_gated_rms_norm_discovers_region() -> None:
    graph = _gated_rms_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    gated = [r for r in regions if r.kind == "region/gated_rms_norm"]
    assert len(gated) == 1
    assert len(gated[0].operation_ids) == 13
    assert gated[0].anchor.component_type == "GatedRMSNorm"


def test_gated_rms_norm_unfused_without_fused_capability() -> None:
    graph = _gated_rms_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 13


def test_gated_rms_norm_fused_lowering_flops_and_allocs() -> None:
    graph = _gated_rms_graph(shape=(4, 8))
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/gated_rms_norm"
    n = 4 * 8
    assert op.forward_flops == 10 * n
    assert op.backward_flops == 14 * n
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("rstd" in aux for aux in op.auxiliary_edges)
    full_rank_aux = [
        aux
        for aux in op.auxiliary_edges
        if aux != f"region:{op.region_id}:rstd"
    ]
    assert full_rank_aux == []


def test_gated_rms_norm_fusion_map_absorbs_all_ops() -> None:
    graph = _gated_rms_graph()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 13
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gated_rms_norm_fla_wins_on_cuda() -> None:
    graph = _gated_rms_graph()
    lowered = lower(graph, _fused_context(hardware="cuda"))
    assert lowered.region_selections[0].chosen.id == "region/gated_rms_norm/fla"


def test_gated_rms_norm_not_double_rmsnorm_silu() -> None:
    graph = _gated_rms_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    region_kinds = [
        step.region.kind for step in plan.steps if step.kind == "region"
    ]
    assert region_kinds == ["region/gated_rms_norm"]


def test_gated_rms_norm_pin_reference() -> None:
    graph = _gated_rms_graph()
    ctx = _fused_context(
        region_implementation_pins=MappingProxyType(
            {"region/gated_rms_norm": "region/gated_rms_norm/reference"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/gated_rms_norm/reference"


@pytest.mark.skip(
    reason=(
        "GatedDeltaNet calls out_norm per head; one provenance envelope spans all "
        "heads with parent reshape gaps — contiguous hybrid discovery is TBD"
    )
)
def test_gated_delta_net_stack_includes_gated_rms_norm_leaf() -> None:
    cfg = GatedDeltaNetConfig(
        hidden_size=32,
        num_qk_heads=2,
        num_v_heads=2,
        head_dim=16,
    )
    graph = compose_graph(
        lambda _ctx: GatedDeltaNet(cfg),
        (Tensor(shape=(4, cfg.hidden_size), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    kinds = {r.kind for r in regions}
    assert "region/gated_rms_norm" in kinds
    assert "region/gated_delta_scan" in kinds
