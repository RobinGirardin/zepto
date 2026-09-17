"""Discovery, lowering, and variant tests for region/gated_grouped_rms_norm."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.gated_grouped_rms_norm import (
    DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
)
from zepto.compose import Tensor, compose_graph
from zepto.modules.gated_grouped_rms_norm import GatedGroupedRMSNorm
from zepto.modules.gated_rms_norm import GatedRMSNorm
from zepto.semantic import ResourceEventKind


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _graph(*, seq: int = 4, hidden: int = 8, groups: int = 2):
    return compose_graph(
        lambda ctx: GatedGroupedRMSNorm(hidden, groups),
        (
            Tensor(shape=(seq, hidden), requires_grad=True),
            Tensor(shape=(seq, hidden), requires_grad=True),
        ),
    )


def test_compose_gated_grouped_rms_norm_discovers_region() -> None:
    graph = _graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    gated = [r for r in regions if r.kind == "region/gated_grouped_rms_norm"]
    assert len(gated) == 1
    assert len(gated[0].operation_ids) == 15
    assert gated[0].anchor.component_type == "GatedGroupedRMSNorm"


def test_gated_grouped_rms_norm_unfused_without_fused_capability() -> None:
    graph = _graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 15


def test_gated_grouped_rms_norm_fused_lowering_flops_and_allocs() -> None:
    graph = _graph(seq=4, hidden=8, groups=2)
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/gated_grouped_rms_norm"
    n = 4 * 8
    assert op.forward_flops == 10 * n
    assert op.backward_flops == 14 * n
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("group_rstd" in aux for aux in op.auxiliary_edges)
    full_rank_aux = [
        aux
        for aux in op.auxiliary_edges
        if aux != f"region:{op.region_id}:group_rstd"
    ]
    assert full_rank_aux == []


def test_gated_grouped_rms_norm_group_rstd_shape() -> None:
    recipe = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE
    assert recipe.group_rstd_shape((4, 8), 2) == (4, 2, 1)
    assert recipe.group_rstd_shape((2, 4, 8), 2) == (2, 4, 2, 1)


def test_gated_grouped_rms_norm_fusion_map_absorbs_all_ops() -> None:
    graph = _graph()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 15
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gated_grouped_rms_norm_not_double_silu_rmsnorm() -> None:
    graph = _graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    region_kinds = [
        step.region.kind for step in plan.steps if step.kind == "region"
    ]
    assert region_kinds == ["region/gated_grouped_rms_norm"]


def test_gated_grouped_rms_norm_pin_reference() -> None:
    graph = _graph()
    ctx = _fused_context(
        region_implementation_pins=MappingProxyType(
            {
                "region/gated_grouped_rms_norm": (
                    "region/gated_grouped_rms_norm/reference"
                )
            }
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/gated_grouped_rms_norm/reference"


def test_gated_grouped_rms_norm_pin_eager_hf() -> None:
    graph = _graph(seq=4, hidden=8, groups=2)
    ctx = _fused_context(
        region_implementation_pins=MappingProxyType(
            {
                "region/gated_grouped_rms_norm": (
                    "region/gated_grouped_rms_norm/eager-hf"
                )
            }
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/gated_grouped_rms_norm/eager-hf"
    n = 4 * 8
    assert lowered.nodes[0].forward_flops == 10 * n + 2 * 4 * 2


def test_gated_grouped_rms_norm_distinct_from_gated_rms_norm() -> None:
    graph = compose_graph(
        lambda ctx: GatedRMSNorm(8),
        (
            Tensor(shape=(4, 8), requires_grad=True),
            Tensor(shape=(4, 8), requires_grad=True),
        ),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    kinds = {r.kind for r in regions}
    assert kinds == {"region/gated_rms_norm"}


def test_gated_grouped_rms_norm_backward_zero_when_inference() -> None:
    graph = compose_graph(
        lambda ctx: GatedGroupedRMSNorm(8, 2),
        (
            Tensor(shape=(4, 8), requires_grad=False),
            Tensor(shape=(4, 8), requires_grad=False),
        ),
    )
    lowered = lower(graph, _fused_context())
    assert lowered.nodes[0].backward_flops == 0
