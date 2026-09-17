"""Discovery, lowering, and variant tests for region/l2_normalize."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.graph import GraphBuilder, Provenance
from zepto.modules.l2_normalize import L2Normalize
from zepto.semantic import Add, Cast, Divide, Multiply, Port, ReduceSum, SquareRoot
from zepto.semantic.metadata import DType
from zepto.semantic import ResourceEventKind


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _build_l2_normalize_chain_graph(
    *,
    shape: tuple[int, int] = (4, 8),
    module_path: tuple[str, ...] = ("Block", "l2_norm"),
    requires_grad: bool = True,
):
    """Build the 7-op decomposed L2-normalize chain (fp32 reduction path)."""
    s, d = shape
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=shape, requires_grad=requires_grad))
    eps = builder.add_input(
        Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)
    )

    def prov(family: str, instance: int) -> Provenance:
        return Provenance(module_path, "L2Normalize", family, instance)

    x_fp32 = builder.add_operation(
        operation_family="cast",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(x,),
        output_tensors=(
            Tensor(shape=shape, dtype=DType.FP32, requires_grad=requires_grad),
        ),
        provenance=prov("cast", 0),
        operation=Cast(to_dtype=DType.FP32),
    )[0]
    squared = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x_fp32, x_fp32),
        output_tensors=(Tensor(shape=shape, dtype=DType.FP32, requires_grad=requires_grad),),
        provenance=prov("multiply", 0),
        operation=Multiply(),
    )[0]
    norm_sq = builder.add_operation(
        operation_family="reduce_sum",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(squared,),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=requires_grad),),
        provenance=prov("reduce_sum", 0),
        operation=ReduceSum(axis=-1, keepdim=True),
    )[0]
    radicand = builder.add_operation(
        operation_family="add",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(norm_sq, eps),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=requires_grad),),
        provenance=prov("add", 0),
        operation=Add(),
    )[0]
    denom = builder.add_operation(
        operation_family="square_root",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(radicand,),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=requires_grad),),
        provenance=prov("square_root", 0),
        operation=SquareRoot(),
    )[0]
    normalized = builder.add_operation(
        operation_family="divide",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x_fp32, denom),
        output_tensors=(Tensor(shape=shape, dtype=DType.FP32, requires_grad=requires_grad),),
        provenance=prov("divide", 0),
        operation=Divide(),
    )[0]
    output = builder.add_operation(
        operation_family="cast",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(normalized,),
        output_tensors=(Tensor(shape=shape, dtype=DType.BF16, requires_grad=requires_grad),),
        provenance=prov("cast", 1),
        operation=Cast(to_dtype=DType.BF16),
    )[0]
    builder.mark_output(output)
    return builder.build(), output


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_discover_l2_normalize_region() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    l2 = [r for r in regions if r.kind == "region/l2_normalize"]
    assert len(l2) == 1
    assert len(l2[0].operation_ids) == 7
    assert l2[0].anchor.component_type == "L2Normalize"


def test_build_plan_fuses_l2_normalize_block() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    region_steps = [step for step in plan.steps if step.kind == "region"]
    assert len(region_steps) == 1
    assert len(region_steps[0].region.operation_ids) == 7


def test_compose_l2_normalize_discovers_region() -> None:
    graph = compose_graph(
        lambda ctx: L2Normalize(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert len([r for r in regions if r.kind == "region/l2_normalize"]) == 1


def test_l2_normalize_unfused_without_fused_capability() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 7


def test_fused_l2_normalize_elides_internal_allocs() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    fused = lower(graph, _fused_context())
    per_op = lower(graph, reference_invocation(), registry=_identity_registry())
    assert len(fused.nodes) == 1
    assert len(fused.fusion_map) == 7
    fused_alloc = sum(
        1
        for op in fused.nodes
        for ev in op.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    per_op_alloc = sum(
        1
        for op in per_op.nodes
        for ev in op.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert fused_alloc < per_op_alloc
    assert fused_alloc == 2  # y + rstd


def test_l2_normalize_forward_flops_3n() -> None:
    graph, _ = _build_l2_normalize_chain_graph(shape=(4, 8))
    lowered = lower(graph, _fused_context())
    assert lowered.nodes[0].forward_flops == 3 * 4 * 8


def test_l2_normalize_backward_flops_4n() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    op = lower(graph, _fused_context()).nodes[0]
    assert op.backward_flops == 4 * 4 * 8
    graph_inf, _ = _build_l2_normalize_chain_graph(requires_grad=False)
    op_inf = lower(graph_inf, _fused_context()).nodes[0]
    assert op_inf.backward_flops == 0


def test_l2_normalize_saves_rstd() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    lowered = lower(graph, _fused_context())
    op = lowered.nodes[0]
    assert any("rstd" in aux for aux in op.auxiliary_edges)
    assert not any("mean" in aux for aux in op.auxiliary_edges)
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1


def test_l2_normalize_pin_reference() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    ctx = _fused_context(
        region_implementation_pins=MappingProxyType(
            {"region/l2_normalize": "region/l2_normalize/reference"}
        )
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/l2_normalize/reference"


def test_l2_normalize_fla_wins_on_generic_hardware() -> None:
    graph, _ = _build_l2_normalize_chain_graph()
    lowered = lower(graph, _fused_context())
    assert lowered.region_selections[0].chosen.id == "region/l2_normalize/fla"
