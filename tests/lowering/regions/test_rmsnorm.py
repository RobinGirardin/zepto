"""Discovery, lowering, and variant tests for region/rmsnorm."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Parameter, Tensor, compose_graph
from zepto.graph import GraphBuilder, Provenance
from zepto.modules.blocks.apertus_decoder_block import ApertusDecoderBlock
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.semantic import (
    Add,
    Cast,
    Divide,
    Multiply,
    ParameterScale,
    Port,
    ReduceSum,
    ResourceEventKind,
    SquareRoot,
    ValueKind,
)
from zepto.semantic.metadata import DType


def _build_rmsnorm_chain_graph(
    *,
    shape: tuple[int, int] = (4, 8),
    module_path: tuple[str, ...] = ("Block", "pre_attn_norm"),
):
    """Build the 9-op decomposed RMSNorm chain (HF fp32 variance path)."""
    s, d = shape
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=shape, requires_grad=True))
    inv_norm = builder.add_input(
        Tensor(shape=(1,), semantic_type="inv_norm_size", requires_grad=False)
    )
    eps = builder.add_input(
        Tensor(shape=(1,), semantic_type="epsilon", requires_grad=False)
    )
    weight = builder.add_parameter(
        Parameter(shape=(d,), semantic_type="weight")
    )

    def prov(family: str, instance: int) -> Provenance:
        return Provenance(module_path, "RMSNorm", family, instance)

    x_fp32 = builder.add_operation(
        operation_family="cast",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(x,),
        output_tensors=(
            Tensor(shape=shape, dtype=DType.FP32, requires_grad=True),
        ),
        provenance=prov("cast", 0),
        operation=Cast(to_dtype=DType.FP32),
    )[0]
    squared = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x_fp32, x_fp32),
        output_tensors=(Tensor(shape=shape, dtype=DType.FP32, requires_grad=True),),
        provenance=prov("multiply", 0),
        operation=Multiply(),
    )[0]
    sum_sq = builder.add_operation(
        operation_family="reduce_sum",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(squared,),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=True),),
        provenance=prov("reduce_sum", 0),
        operation=ReduceSum(axis=-1, keepdim=True),
    )[0]
    variance = builder.add_operation(
        operation_family="divide",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(sum_sq, inv_norm),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=True),),
        provenance=prov("divide", 0),
        operation=Divide(),
    )[0]
    var_eps = builder.add_operation(
        operation_family="add",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(variance, eps),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=True),),
        provenance=prov("add", 0),
        operation=Add(),
    )[0]
    denom = builder.add_operation(
        operation_family="square_root",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(var_eps,),
        output_tensors=(Tensor(shape=(s, 1), dtype=DType.FP32, requires_grad=True),),
        provenance=prov("square_root", 0),
        operation=SquareRoot(),
    )[0]
    normalized = builder.add_operation(
        operation_family="divide",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x_fp32, denom),
        output_tensors=(Tensor(shape=shape, dtype=DType.FP32, requires_grad=True),),
        provenance=prov("divide", 1),
        operation=Divide(),
    )[0]
    normalized_act = builder.add_operation(
        operation_family="cast",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(normalized,),
        output_tensors=(Tensor(shape=shape, dtype=DType.BF16, requires_grad=True),),
        provenance=prov("cast", 1),
        operation=Cast(to_dtype=DType.BF16),
    )[0]
    output = builder.add_operation(
        operation_family="parameter_scale",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        parameter_ports=(Port("weight", value_kind=ValueKind.PARAMETER),),
        input_edges=(normalized_act,),
        parameter_ids=(weight,),
        output_tensors=(Tensor(shape=shape, dtype=DType.BF16, requires_grad=True),),
        provenance=prov("parameter_scale", 0),
        operation=ParameterScale(),
    )[0]
    builder.mark_output(output)
    return builder.build(), output


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_discover_hybrid_rmsnorm_region() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    rms = [r for r in regions if r.kind == "region/rmsnorm"]
    assert len(rms) == 1
    assert len(rms[0].operation_ids) == 9
    assert rms[0].anchor.component_type == "RMSNorm"


def test_build_plan_fuses_rmsnorm_block() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    region_steps = [step for step in plan.steps if step.kind == "region"]
    assert len(region_steps) == 1
    assert len(region_steps[0].region.operation_ids) == 9


def test_fused_rmsnorm_elides_internal_allocs_vs_identity_sum() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    fused = lower(graph, reference_invocation())
    per_op = lower(graph, reference_invocation(), registry=_identity_registry())
    assert len(fused.nodes) == 1
    assert len(fused.fusion_map) == 9
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
    assert fused_alloc == 2  # y + rstd (default liger on generic hardware)


def test_rmsnorm_forward_flops_4n() -> None:
    graph, _ = _build_rmsnorm_chain_graph(shape=(4, 8))
    lowered = lower(graph, reference_invocation())
    assert lowered.nodes[0].forward_flops == 4 * 4 * 8


def test_rmsnorm_saves_rstd_not_mean() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    assert any("rstd" in aux for aux in op.auxiliary_edges)
    assert not any("mean" in aux for aux in op.auxiliary_edges)


def test_rmsnorm_pin_reference() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    ctx = reference_invocation(
        region_implementation_pins=MappingProxyType(
            {"region/rmsnorm": "region/rmsnorm/reference"}
        )
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/rmsnorm/reference"


def test_rmsnorm_pin_liger() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    ctx = reference_invocation(
        region_implementation_pins=MappingProxyType(
            {"region/rmsnorm": "region/rmsnorm/liger"}
        )
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/rmsnorm/liger"


def test_rmsnorm_hub_xpu_requires_hardware() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    ctx = reference_invocation(
        hardware="xpu",
        region_implementation_pins=MappingProxyType(
            {"region/rmsnorm": "region/rmsnorm/hub-xpu"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/rmsnorm/hub-xpu"


def test_rmsnorm_hub_mps_single_output_alloc() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    ctx = reference_invocation(
        hardware="mps",
        region_implementation_pins=MappingProxyType(
            {"region/rmsnorm": "region/rmsnorm/hub-mps"}
        ),
    )
    lowered = lower(graph, ctx)
    op = lowered.nodes[0]
    assert op.implementation == "region/rmsnorm/hub-mps"
    assert op.auxiliary_edges == ()
    allocs = [
        ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE
    ]
    assert len(allocs) == 1
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert saves == []


def test_rmsnorm_liger_wins_on_generic_hardware() -> None:
    graph, _ = _build_rmsnorm_chain_graph()
    lowered = lower(graph, reference_invocation())
    assert lowered.region_selections[0].chosen.id == "region/rmsnorm/liger"


def test_compose_rmsnorm_discovers_region() -> None:
    graph = compose_graph(
        lambda ctx: RMSNorm(8),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert len([r for r in regions if r.kind == "region/rmsnorm"]) == 1


def test_rmsnorm_batched_flops_scale() -> None:
    from tests.lowering.regions._batch_helpers import assert_flops_scales

    single = lower(
        compose_graph(
            lambda ctx: RMSNorm(8),
            (Tensor(shape=(4, 8), requires_grad=True),),
        ),
        reference_invocation(),
    )
    batched = lower(
        compose_graph(
            lambda ctx: RMSNorm(8),
            (Tensor(shape=(2, 4, 8), requires_grad=True),),
        ),
        reference_invocation(),
    )
    rms_single = [n for n in single.nodes if n.implementation.startswith("region/rmsnorm")][0]
    rms_batched = [n for n in batched.nodes if n.implementation.startswith("region/rmsnorm")][0]
    assert_flops_scales(rms_batched.forward_flops, rms_single.forward_flops, 2)


def test_apertus_decoder_block_discovers_four_rmsnorm_regions() -> None:
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    seq_len = 16
    graph = compose_graph(
        lambda ctx: ApertusDecoderBlock(
            hidden_size=hidden_size,
            intermediate_size=256,
            num_heads=num_heads,
            num_kv_heads=2,
            qk_norm=True,
        ),
        (
            Tensor(shape=(seq_len, hidden_size), requires_grad=True),
            Tensor(shape=(1, seq_len, seq_len)),
            Tensor(shape=(seq_len, head_dim)),
            Tensor(shape=(seq_len, head_dim)),
        ),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    rms = [r for r in regions if r.kind == "region/rmsnorm"]
    assert len(rms) == 4
