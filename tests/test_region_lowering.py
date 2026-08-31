"""Tests for module-aware region lowering."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from zepto.analysis import (
    Region,
    discover_regions,
    lower,
    reference_invocation,
)
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.analysis.lowering.plan import resolve_overlaps
from zepto.compose import Compose, Module, Parameter, Tensor, compose_graph
from zepto.graph import GraphBuilder, Provenance
from zepto.semantic import (
    Add,
    Divide,
    Identity,
    Maximum,
    Multiply,
    Port,
    ReduceSum,
    ResourceEventKind,
    SquareRoot,
    Subtract,
)
from zepto.modules.layer_norm import LayerNorm
from zepto.modules.linear import Linear
from zepto.modules.relu import ReLU


def _build_layernorm_chain_graph(
    *,
    module_path: tuple[str, ...] = ("Block", "norm"),
    component_type: str = "LayerNorm",
):
    """Build a six-op decomposed layer-norm chain for hybrid discovery tests."""
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(4, 8), requires_grad=True))
    eps = builder.add_input(
        Tensor(shape=(1,), semantic_type="constant", requires_grad=False)
    )

    def prov(family: str, instance: int) -> Provenance:
        return Provenance(module_path, component_type, family, instance)

    mean = builder.add_operation(
        operation_family="reduce_sum",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(x,),
        output_tensors=(Tensor(shape=(1, 8), requires_grad=True),),
        provenance=prov("reduce_sum", 0),
        operation=ReduceSum(axis=0, keepdim=True),
    )[0]
    centered = builder.add_operation(
        operation_family="subtract",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x, mean),
        output_tensors=(Tensor(shape=(4, 8), requires_grad=True),),
        provenance=prov("subtract", 0),
        operation=Subtract(),
    )[0]
    squared = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(centered, centered),
        output_tensors=(Tensor(shape=(4, 8), requires_grad=True),),
        provenance=prov("multiply", 0),
        operation=Multiply(),
    )[0]
    var_eps = builder.add_operation(
        operation_family="add",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(squared, eps),
        output_tensors=(Tensor(shape=(4, 8), requires_grad=True),),
        provenance=prov("add", 0),
        operation=Add(),
    )[0]
    std = builder.add_operation(
        operation_family="square_root",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(var_eps,),
        output_tensors=(Tensor(shape=(4, 8), requires_grad=True),),
        provenance=prov("square_root", 0),
        operation=SquareRoot(),
    )[0]
    normalized = builder.add_operation(
        operation_family="divide",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(centered, std),
        output_tensors=(Tensor(shape=(4, 8), requires_grad=True),),
        provenance=prov("divide", 0),
        operation=Divide(),
    )[0]
    builder.mark_output(normalized)
    return builder.build(), normalized


def test_discover_provenance_region_groups_layernorm_ops() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()

    regions = discover_regions(graph, ctx, registry)
    layernorm = [region for region in regions if region.kind == "region/layernorm"]
    assert len(layernorm) == 1
    assert len(layernorm[0].operation_ids) == 6
    assert layernorm[0].anchor.component_type == "LayerNorm"


def test_discover_pattern_region_finds_relu() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(4,), requires_grad=True))
    zero = builder.add_input(
        Tensor(shape=(4,), semantic_type="constant_zero", requires_grad=False)
    )
    builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, zero),
        output_tensors=(Tensor(shape=(4,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    relu_regions = [region for region in regions if region.kind == "region/relu"]
    assert len(relu_regions) == 1
    assert len(relu_regions[0].operation_ids) == 1


def test_build_plan_fuses_contiguous_layernorm_block() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()

    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)

    region_steps = [step for step in plan.steps if step.kind == "region"]
    assert len(region_steps) == 1
    assert region_steps[0].region is not None
    assert len(region_steps[0].region.operation_ids) == 6
    assert len(plan.steps) == 1


def register_identity_only(registry: LoweringRegistry) -> None:
    register_identity_defaults(registry)


def test_build_plan_fallback_when_no_region_impl() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_identity_only(registry)
    ctx = reference_invocation()

    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)

    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 6


def test_overlap_resolution_prefers_larger_region() -> None:
    graph, _output = _build_layernorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()

    regions = discover_regions(graph, ctx, registry)
    small = Region(
        id="small",
        kind="region/relu",
        anchor=regions[0].anchor,
        operation_ids=(graph.node_order[0],),
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
    lowered = lower(graph, reference_invocation())

    assert len(lowered.nodes) == 1
    assert len(lowered.fusion_map) == 6
    assert len(lowered.node_map) == 6


def test_fusion_map_maps_absorbed_ops_to_region_lowered_op() -> None:
    graph, _output = _build_layernorm_chain_graph()
    lowered = lower(graph, reference_invocation())

    region_op_id = lowered.nodes[0].id
    for structural_id, lowered_id in lowered.fusion_map.items():
        assert lowered_id == region_op_id
        assert lowered.node_map[structural_id] == region_op_id


def test_module_path_on_lowered_operation_from_region() -> None:
    graph, _output = _build_layernorm_chain_graph()
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]

    assert op.module_path == ("Block", "norm")
    assert op.component_type == "LayerNorm"
    assert op.region_id is not None


class _ManyOpModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        y = Identity()(x)
        z = Identity()(y)
        return Identity()(z)  # type: ignore[return-value]


def test_mha_without_fusion_lowers_per_op_only() -> None:
    graph = compose_graph(_ManyOpModule(), (Tensor(shape=(4,)),))
    lowered = lower(graph, reference_invocation())

    assert len(lowered.nodes) == 3
    assert not lowered.fusion_map
    assert not lowered.region_map


def test_region_pin_selects_descriptor() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    zero = builder.add_input(Tensor(shape=(2,), semantic_type="constant_zero"))
    builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, zero),
        output_tensors=(Tensor(shape=(2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    ctx = reference_invocation(
        region_implementation_pins=MappingProxyType({"region/relu": "region/relu"})
    )
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].reason == "region_pinned"
    assert lowered.nodes[0].implementation == "region/relu"


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_only(registry)
    return registry


def test_fused_layernorm_elides_internal_allocations_vs_identity_sum() -> None:
    graph, _output = _build_layernorm_chain_graph()
    fused = lower(graph, reference_invocation())
    per_op = lower(graph, reference_invocation(), registry=_identity_registry())

    fused_alloc = sum(
        1
        for op in fused.nodes
        for event in op.resource_events
        if event.kind is ResourceEventKind.ALLOCATE
    )
    per_op_alloc = sum(
        1
        for op in per_op.nodes
        for event in op.resource_events
        if event.kind is ResourceEventKind.ALLOCATE
    )
    assert fused_alloc < per_op_alloc


def test_layernorm_builds_decomposed_graph() -> None:
    graph = compose_graph(lambda ctx: LayerNorm(8), (Tensor(shape=(4, 8)),))
    families = [graph.node(op_id).operation_family for op_id in graph.node_order]
    assert "reduce_sum" in families
    assert "parameter_scale" in families
    assert "parameter_bias" in families
    assert len(graph.node_order) > 1


def test_registered_module_name_in_provenance_path() -> None:
    class Block(Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = LayerNorm(8)

        def forward(self, x: Tensor) -> Tensor:
            return self.norm(x)  # type: ignore[return-value]

    graph = compose_graph(lambda ctx: Block(), (Tensor(shape=(4, 8)),))
    op = graph.node(graph.node_order[0])
    assert op.provenance.module_path == ("Block", "norm")
    assert op.provenance.component_type == "LayerNorm"
    assert op.operation_family == "reduce_sum"


def test_linear_provenance_region_fuses_single_op() -> None:
    graph = compose_graph(lambda ctx: Linear(8, 4), (Tensor(shape=(2, 8)),))
    lowered = lower(graph, reference_invocation())

    assert len(lowered.nodes) == 1
    assert lowered.nodes[0].implementation == "region/linear"
    assert lowered.nodes[0].component_type == "Linear"


def test_hybrid_skips_when_pattern_fails() -> None:
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(4,)))
    builder.add_operation(
        operation_family="identity",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(x,),
        output_tensors=(Tensor(shape=(4,)),),
        provenance=Provenance(("Block", "norm"), "LayerNorm", "identity", 0),
        operation=Identity(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert not any(region.kind == "region/layernorm" for region in regions)


def test_relu_module_builds_maximum_graph() -> None:
    graph = compose_graph(lambda ctx: ReLU(), (Tensor(shape=(4, 8), requires_grad=True),))
    op = graph.node(graph.node_order[0])
    assert op.operation_family == "maximum"
    assert op.provenance.component_type == "ReLU"


def test_relu_lowers_to_region() -> None:
    graph = compose_graph(lambda ctx: ReLU(), (Tensor(shape=(4,), requires_grad=True),))
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    assert lowered.nodes[0].implementation == "region/relu"


def test_auto_setattr_registers_submodule() -> None:
    with Compose():
        class Block(Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = LayerNorm(8)

            def forward(self, x: Tensor) -> Tensor:
                return self.norm(x)  # type: ignore[return-value]

        block = Block()
        assert "norm" in block._modules
        assert block.norm._registered_name == "norm"


def test_auto_setattr_registers_parameter() -> None:
    graph = compose_graph(lambda ctx: Linear(4, 8), (Tensor(shape=(2, 4)),))
    assert len(graph.parameters) == 1


def test_auto_setattr_registers_buffer() -> None:
    class BufferModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self._eps = Tensor(
                shape=(1,), semantic_type="epsilon", requires_grad=False
            )

        def forward(self, x: Tensor) -> Tensor:
            return x  # type: ignore[return-value]

    graph = compose_graph(lambda ctx: BufferModule(), (Tensor(shape=(4, 8)),))
    assert len(graph.inputs) == 2


def test_duplicate_module_assignment_raises() -> None:
    with Compose():
        class Block(Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = LayerNorm(8)

        block = Block()
        with pytest.raises(ValueError, match="duplicate module member"):
            block.norm = LayerNorm(8)


def test_duplicate_parameter_assignment_raises() -> None:
    with Compose():
        class ParamModule(Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = Parameter(shape=(4, 4), semantic_type="weight")

        module = ParamModule()
        with pytest.raises(ValueError, match="duplicate module member"):
            module.weight = Parameter(shape=(4, 4), semantic_type="weight")
