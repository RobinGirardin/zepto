"""Discovery, lowering, and variant tests for region/depthwise_causal_conv1d."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.horizon.state import Conv1DState
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.depthwise_causal_conv1d import DepthwiseCausalConv1dRecipe
from zepto.compose import Tensor, compose_graph
from zepto.modules.mixers.depthwise_causal_conv1d import DepthwiseCausalConv1d
from zepto.semantic import ResourceEventKind
from zepto.semantic.metadata import DType


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _prefill_graph(*, seq_len: int = 4, channels: int = 8, **module_kw):
    return compose_graph(
        lambda ctx: DepthwiseCausalConv1d(
            channels=channels,
            kernel_size=4,
            activation="silu",
            **module_kw,
        ),
        (Tensor(shape=(seq_len, channels), requires_grad=True),),
    )


def test_compose_depthwise_causal_conv1d_discovers_region() -> None:
    graph = _prefill_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    conv = [r for r in regions if r.kind == "region/depthwise_causal_conv1d"]
    assert len(conv) == 1
    assert conv[0].anchor.component_type == "DepthwiseCausalConv1d"
    assert conv[0].matcher_id == "prov-depthwise-causal-conv1d"


def test_depthwise_causal_conv1d_unfused_without_fused_capability() -> None:
    graph = _prefill_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) > 1


def test_depthwise_causal_conv1d_fused_lowering_flops_and_allocs() -> None:
    seq_len, channels = 4, 8
    graph = _prefill_graph(seq_len=seq_len, channels=channels)
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/depthwise_causal_conv1d/decomposed"
    assert op.forward_flops == 12 * seq_len * channels
    assert op.backward_flops == 22 * seq_len * channels
    alloc_ids = [ev.value for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(alloc_ids) >= 2
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("pre_activation" in aux for aux in op.auxiliary_edges)


def test_depthwise_causal_conv1d_fusion_map_absorbs_module_ops() -> None:
    graph = _prefill_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    n_ops = len(regions[0].operation_ids)
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == n_ops
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_depthwise_causal_conv1d_cuda_wins_on_cuda_hardware() -> None:
    graph = _prefill_graph()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert (
        lowered.region_selections[0].chosen.id
        == "region/depthwise_causal_conv1d/cuda"
    )


def test_depthwise_causal_conv1d_decode_single_step() -> None:
    channels, kernel_size = 8, 4
    graph = compose_graph(
        lambda ctx: DepthwiseCausalConv1d(
            channels=channels,
            kernel_size=kernel_size,
            activation="silu",
        ),
        (
            Tensor(shape=(1, channels), requires_grad=True),
            Tensor(shape=(channels, kernel_size), requires_grad=False),
        ),
    )
    conv_state = Conv1DState(
        layer_index=0,
        channels=channels,
        kernel_size=kernel_size,
        dtype=DType.FP32,
    )
    ctx = _fused_context(state=(("conv_state:0", conv_state),))
    lowered = lower(graph, ctx)
    op = lowered.nodes[0]
    assert op.forward_flops == 12 * channels
    assert any("conv_state" in aux for aux in op.auxiliary_edges)
    persists = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.PERSIST]
    assert persists
    assert len(lowered.state_port_events) == 1


def test_depthwise_causal_conv1d_bias_and_no_activation_recipe() -> None:
    recipe = DepthwiseCausalConv1dRecipe(
        kernel_size=4,
        use_bias=True,
        activation="none",
    )
    assert recipe.forward_flops_per_element() == 8
    assert recipe.backward_flops_per_element() == 14


def test_depthwise_causal_conv1d_batched_prefill_flops_scale() -> None:
    from tests.lowering.regions._batch_helpers import assert_flops_scales

    batch, seq_len, channels = 2, 4, 8
    single = lower(_prefill_graph(seq_len=seq_len, channels=channels), _fused_context())
    batched = lower(
        compose_graph(
            lambda ctx: DepthwiseCausalConv1d(
                channels=channels, kernel_size=4, activation="silu"
            ),
            (Tensor(shape=(batch, seq_len, channels), requires_grad=True),),
        ),
        _fused_context(),
    )
    assert_flops_scales(
        batched.nodes[0].forward_flops, single.nodes[0].forward_flops, batch
    )


def test_depthwise_causal_conv1d_batched_decode() -> None:
    batch, channels = 2, 8
    graph = compose_graph(
        lambda ctx: DepthwiseCausalConv1d(
            channels=channels,
            kernel_size=4,
            activation="silu",
        ),
        (Tensor(shape=(batch, 1, channels), requires_grad=True),),
    )
    lowered = lower(graph, _fused_context())
    assert lowered.nodes[0].forward_flops == 12 * batch * channels
