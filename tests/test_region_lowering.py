"""Tests for module-aware region lowering."""

from __future__ import annotations

from types import MappingProxyType

from zepto.core import (
    Add,
    Divide,
    Identity,
    LoweringError,
    Maximum,
    Multiply,
    PortSpec,
    Provenance,
    ReduceSum,
    ResourceEventKind,
    SquareRoot,
    StructuralGraphBuilder,
    Subtract,
    ValueMetadata,
    build_graph,
    lower,
    reference_context,
)
from zepto.core.composition import GraphCompositionContext, Module
from zepto.core.functional import identity
from zepto.core.lowering import (
    LoweringRegistry,
    build_lowering_plan,
    discover_regions,
)
from zepto.core.lowering.implementations import register_defaults
from zepto.core.lowering.plan import resolve_overlaps
from zepto.core.lowering.region import (
    StructuralRegion,
)
from zepto.modules.layer_norm import LayerNorm
from zepto.modules.linear import Linear


def _layernorm_provenance(path: tuple[str, ...] = ("Block", "norm")) -> Provenance:
    return Provenance(path, "LayerNorm", "reduce_sum", 0)


def _build_layernorm_chain_graph(
    *,
    module_path: tuple[str, ...] = ("Block", "norm"),
    component_type: str = "LayerNorm",
) -> tuple:
    """Build a six-op decomposed layer-norm chain for hybrid discovery tests."""
    builder = StructuralGraphBuilder()
    x = builder.add_input(ValueMetadata((4, 8), requires_grad=True))
    eps = builder.add_input(
        ValueMetadata((1,), semantic_type="constant", requires_grad=False)
    )

    def prov(family: str, instance: int) -> Provenance:
        return Provenance(module_path, component_type, family, instance)

    mean = builder.add_operation(
        operation_family="reduce_sum",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(x,),
        output_metadata=(ValueMetadata((1, 8), requires_grad=True),),
        provenance=prov("reduce_sum", 0),
        operation=ReduceSum(axis=0, keepdim=True),
    )[0]
    centered = builder.add_operation(
        operation_family="subtract",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(x, mean),
        output_metadata=(ValueMetadata((4, 8), requires_grad=True),),
        provenance=prov("subtract", 0),
        operation=Subtract(),
    )[0]
    squared = builder.add_operation(
        operation_family="multiply",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(centered, centered),
        output_metadata=(ValueMetadata((4, 8), requires_grad=True),),
        provenance=prov("multiply", 0),
        operation=Multiply(),
    )[0]
    var_eps = builder.add_operation(
        operation_family="add",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(squared, eps),
        output_metadata=(ValueMetadata((4, 8), requires_grad=True),),
        provenance=prov("add", 0),
        operation=Add(),
    )[0]
    std = builder.add_operation(
        operation_family="square_root",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(var_eps,),
        output_metadata=(ValueMetadata((4, 8), requires_grad=True),),
        provenance=prov("square_root", 0),
        operation=SquareRoot(),
    )[0]
    normalized = builder.add_operation(
        operation_family="divide",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(centered, std),
        output_metadata=(ValueMetadata((4, 8), requires_grad=True),),
        provenance=prov("divide", 0),
        operation=Divide(),
    )[0]
    builder.mark_output(normalized)
    return builder.build(), normalized


def test_discover_provenance_region_groups_layernorm_ops() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_context()

    regions = discover_regions(graph, ctx, registry)
    layernorm = [region for region in regions if region.kind == "region/layernorm"]
    assert len(layernorm) == 1
    assert len(layernorm[0].operation_ids) == 6
    assert layernorm[0].anchor.component_type == "LayerNorm"


def test_discover_pattern_region_finds_relu() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(ValueMetadata((4,), requires_grad=True))
    zero = builder.add_input(
        ValueMetadata((4,), semantic_type="constant_zero", requires_grad=False)
    )
    builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, zero),
        output_metadata=(ValueMetadata((4,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_context(), registry)
    relu_regions = [region for region in regions if region.kind == "region/relu"]
    assert len(relu_regions) == 1
    assert len(relu_regions[0].operation_ids) == 1


def test_build_plan_fuses_contiguous_layernorm_block() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_context()

    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)

    region_steps = [step for step in plan.steps if step.kind == "region"]
    assert len(region_steps) == 1
    assert len(region_steps[0].region.operation_ids) == 6
    assert len(plan.steps) == 1


def test_build_plan_fallback_when_no_region_impl() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_identity_only(registry)
    ctx = reference_context()

    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)

    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 6


def register_identity_only(registry: LoweringRegistry) -> None:
    from zepto.core.lowering.implementations import register_identity_defaults

    register_identity_defaults(registry)


def test_overlap_resolution_prefers_larger_region() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_context()

    regions = discover_regions(graph, ctx, registry)
    small = StructuralRegion(
        id="small",
        kind="region/relu",
        anchor=regions[0].anchor,
        operation_ids=(graph.operations[0],),
        boundary_inputs=(),
        boundary_outputs=(),
        parameter_ids=(),
        matcher_id="test",
    )
    winners = resolve_overlaps(graph, (*regions, small), ctx, registry)
    assert any(len(region.operation_ids) == 6 for region in winners)
    assert not any(region.id == "small" for region in winners)


def test_lower_region_marks_ops_consumed_no_double_lowering() -> None:
    graph, _output = _build_layernorm_chain_graph()
    lowered = lower(graph, reference_context())

    assert len(lowered.operations) == 1
    assert len(lowered.fusion_map) == 6
    assert len(lowered.operation_map) == 6


def test_fusion_map_maps_absorbed_ops_to_region_lowered_op() -> None:
    graph, _output = _build_layernorm_chain_graph()
    lowered = lower(graph, reference_context())

    region_op_id = lowered.operations[0].id
    for structural_id, lowered_id in lowered.fusion_map.items():
        assert lowered_id == region_op_id
        assert lowered.operation_map[structural_id] == region_op_id


def test_module_path_on_lowered_operation_from_region() -> None:
    graph, _output = _build_layernorm_chain_graph()
    lowered = lower(graph, reference_context())
    op = lowered.operations[0]

    assert op.module_path == ("Block", "norm")
    assert op.component_type == "LayerNorm"
    assert op.region_id is not None


class _ManyOpModule(Module):
    def forward(self, x):
        y = identity(x)
        z = identity(y)
        return identity(z)


def test_mha_without_fusion_lowers_per_op_only() -> None:
    graph = build_graph(_ManyOpModule(), (ValueMetadata((4,)),))
    lowered = lower(graph, reference_context())

    assert len(lowered.operations) == 3
    assert not lowered.fusion_map
    assert not lowered.region_map


def test_region_pin_selects_descriptor() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(ValueMetadata((2,), requires_grad=True))
    zero = builder.add_input(
        ValueMetadata((2,), semantic_type="constant_zero")
    )
    builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, zero),
        output_metadata=(ValueMetadata((2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    ctx = reference_context(
        region_implementation_pins=MappingProxyType(
            {"region/relu": "region/relu"}
        )
    )
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].reason == "region_pinned"
    assert lowered.operations[0].implementation == "region/relu"


def test_fused_layernorm_elides_internal_allocations_vs_identity_sum() -> None:
    graph, _output = _build_layernorm_chain_graph()
    fused = lower(graph, reference_context())
    per_op = lower(graph, reference_context(), registry=_identity_registry())

    fused_alloc = sum(
        1
        for op in fused.operations
        for event in op.resource_events
        if event.kind is ResourceEventKind.ALLOCATE
    )
    per_op_alloc = sum(
        1
        for op in per_op.operations
        for event in op.resource_events
        if event.kind is ResourceEventKind.ALLOCATE
    )
    assert fused_alloc < per_op_alloc


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_only(registry)
    return registry


def test_layernorm_builds_decomposed_graph() -> None:
    graph = build_graph(LayerNorm(8), (ValueMetadata((4, 8)),))
    families = [graph.operation(op_id).operation_family for op_id in graph.operations]
    assert "reduce_sum" in families
    assert "parameter_scale" in families
    assert "parameter_bias" in families
    assert len(graph.operations) > 1


def test_registered_module_name_in_provenance_path() -> None:
    class Block(Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = LayerNorm(8)

        def forward(self, x):
            return self.norm(x)

    graph = build_graph(Block(), (ValueMetadata((4, 8)),))
    op = graph.operation(graph.operations[0])
    assert op.provenance.module_path == ("Block", "norm")
    assert op.provenance.component_type == "LayerNorm"
    assert op.operation_family == "reduce_sum"


def test_linear_provenance_region_fuses_single_op() -> None:
    def factory(ctx: GraphCompositionContext) -> Module:
        return Linear(8, 4)

    graph = build_graph(factory, (ValueMetadata((2, 8)),))
    lowered = lower(graph, reference_context())

    assert len(lowered.operations) == 1
    assert lowered.operations[0].implementation == "region/linear"
    assert lowered.operations[0].component_type == "Linear"


def test_hybrid_skips_when_pattern_fails() -> None:
    builder = StructuralGraphBuilder()
    x = builder.add_input(ValueMetadata((4,)))
    builder.add_operation(
        operation_family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(x,),
        output_metadata=(ValueMetadata((4,)),),
        provenance=Provenance(("Block", "norm"), "LayerNorm", "identity", 0),
        operation=Identity(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_context(), registry)
    assert not any(region.kind == "region/layernorm" for region in regions)


def test_relu_module_builds_maximum_graph() -> None:
    from zepto.modules.relu import ReLU

    graph = build_graph(ReLU(), (ValueMetadata((4, 8), requires_grad=True),))
    op = graph.operation(graph.operations[0])
    assert op.operation_family == "maximum"
    assert op.provenance.component_type == "ReLU"


def test_functional_relu_delegates_to_module() -> None:
    from zepto.core.functional import relu

    def factory(_ctx: GraphCompositionContext) -> Module:
        class ReluWrapper(Module):
            def forward(self, x):
                return relu(x)

        return ReluWrapper()

    graph = build_graph(factory, (ValueMetadata((4,), requires_grad=True),))
    op = graph.operation(graph.operations[0])
    assert op.operation_family == "maximum"


def test_relu_lowers_to_region() -> None:
    from zepto.modules.relu import ReLU

    graph = build_graph(ReLU(), (ValueMetadata((4,), requires_grad=True),))
    lowered = lower(graph, reference_context())
    assert len(lowered.operations) == 1
    assert lowered.operations[0].implementation == "region/relu"


def test_auto_setattr_registers_submodule() -> None:
    class Block(Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = LayerNorm(8)

        def forward(self, x):
            return self.norm(x)

    block = Block()
    assert "norm" in block._modules
    assert block.norm._registered_name == "norm"


def test_auto_setattr_registers_parameter() -> None:
    def factory(ctx: GraphCompositionContext) -> Linear:
        return Linear(4, 8)

    graph = build_graph(factory, (ValueMetadata((2, 4)),))
    assert len(graph.parameters) == 1


def test_duplicate_module_assignment_raises() -> None:
    import pytest

    class Block(Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = LayerNorm(8)

    block = Block()
    with pytest.raises(ValueError, match="duplicate module member"):
        block.norm = LayerNorm(8)


def test_duplicate_parameter_assignment_raises() -> None:
    import pytest
    from zepto.core.composition import GraphCompositionContext

    with GraphCompositionContext():
        class ParamModule(Module):
            def __init__(self) -> None:
                super().__init__()
                ctx = GraphCompositionContext.current()
                assert ctx is not None
                self.weight = ctx.parameter(ValueMetadata((4, 4), semantic_type="weight"))

        module = ParamModule()
        with pytest.raises(ValueError, match="duplicate module member"):
            module.weight = GraphCompositionContext.current().parameter(  # type: ignore[union-attr]
                ValueMetadata((4, 4), semantic_type="weight")
            )
