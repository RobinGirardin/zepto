"""Discovery, lowering, and variant tests for region/gated_delta_net."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from zepto.analysis import (
    ConvStateConfig,
    HorizonSpec,
    RecurrentStateConfig,
    discover_regions,
    estimate_horizon,
    lower,
    reference_invocation,
)
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.plan import resolve_overlaps
from zepto.analysis.lowering.recipes import (
    DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    DEFAULT_GATED_DELTA_NET_RECIPE,
    DEFAULT_GATED_DELTA_SCAN_RECIPE,
    DEFAULT_GATED_RMS_NORM_RECIPE,
    DEFAULT_L2_NORMALIZE_RECIPE,
    FLA_LAYER_PARITY_GATED_DELTA_NET_RECIPE,
    GatedDeltaNetRecipe,
)
from zepto.compose import Tensor, compose_graph
from modules.mixers.gated_delta_net import GatedDeltaNet
from modules.mixers.mixer_config import GatedDeltaNetConfig
from zepto.semantic import ResourceEventKind
from zepto.semantic.metadata import DType


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _mini_cfg() -> GatedDeltaNetConfig:
    return GatedDeltaNetConfig(
        hidden_size=32,
        num_qk_heads=2,
        num_v_heads=2,
        head_dim=16,
    )


def _gdn_graph(
    *,
    seq_len: int = 4,
    cfg: GatedDeltaNetConfig | None = None,
    requires_grad: bool = True,
    with_states: bool = False,
):
    cfg = cfg or _mini_cfg()
    inputs: tuple[Tensor, ...] = (
        Tensor(shape=(seq_len, cfg.hidden_size), requires_grad=requires_grad),
    )
    if with_states:
        conv_c = 2 * cfg.qk_proj_size + cfg.v_proj_size
        inputs = (
            *inputs,
            Tensor(shape=(conv_c, cfg.conv_kernel_size), requires_grad=False),
            Tensor(
                shape=(cfg.num_v_heads, cfg.head_dim, cfg.head_dim),
                requires_grad=False,
            ),
        )
    return compose_graph(lambda _ctx: GatedDeltaNet(cfg), inputs)


def _leaf_sum_flops(cfg: GatedDeltaNetConfig, *, seq_len: int, recipe: GatedDeltaNetRecipe):
    s = seq_len
    d = cfg.hidden_size
    h_k, h_v, d_k = cfg.num_qk_heads, cfg.num_v_heads, cfg.head_dim
    d_v = cfg.head_dim
    d_q, d_v_proj = cfg.qk_proj_size, cfg.v_proj_size
    channels = 2 * d_q + d_v_proj
    linear = 4 * s * d * (d_q + d_v_proj) + 4 * s * d * h_v + 2 * s * d_v_proj * d
    conv = DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE.forward_flops(s * channels)
    l2 = 2 * DEFAULT_L2_NORMALIZE_RECIPE.forward_flops(s * h_k * d_k)
    glue = s * h_v * (d_k + 11)
    scan = DEFAULT_GATED_DELTA_SCAN_RECIPE.forward_flops(
        seq_len=s, num_heads=h_v, key_dim=d_k, value_dim=d_v
    )
    grms = DEFAULT_GATED_RMS_NORM_RECIPE.forward_flops(s * h_v * d_v)
    total = linear + conv + l2 + glue + scan + grms
    if recipe.use_qk_l2norm_in_kernel:
        total -= 6 * s * h_k * d_k
    if recipe.use_gate_in_kernel:
        total -= 10 * s * h_v * d_v
    return total


def test_compose_gated_delta_net_discovers_block_region() -> None:
    graph = _gdn_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    blocks = [r for r in regions if r.kind == "region/gated_delta_net"]
    assert len(blocks) == 1
    assert blocks[0].anchor.component_type == "GatedDeltaNet"
    assert blocks[0].matcher_id == "prov-gated-delta-net"
    assert len(blocks[0].operation_ids) == len(graph.node_order)


def test_gated_delta_net_unfused_without_fused_capability() -> None:
    graph = _gdn_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) > 1


def test_gated_delta_net_composed_tier_a_flops_match_leaf_sum() -> None:
    cfg = _mini_cfg()
    recipe = DEFAULT_GATED_DELTA_NET_RECIPE
    seq_len = 4
    assert recipe.forward_flops(cfg, seq_len=seq_len) == _leaf_sum_flops(
        cfg, seq_len=seq_len, recipe=recipe
    )
    graph = _gdn_graph(seq_len=seq_len, cfg=cfg)
    lowered = lower(graph, _fused_context())
    assert lowered.nodes[0].forward_flops == recipe.forward_flops(cfg, seq_len=seq_len)


def test_gated_delta_net_subsumes_child_regions() -> None:
    graph = _gdn_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context()
    regions = discover_regions(graph, ctx, registry)
    winners = resolve_overlaps(graph, regions, ctx, registry)
    block_winners = [r for r in winners if r.kind == "region/gated_delta_net"]
    assert len(block_winners) == 1
    child_kinds = {
        "region/gated_delta_scan",
        "region/l2_normalize",
        "region/gated_rms_norm",
        "region/depthwise_causal_conv1d",
    }
    for region in winners:
        if region.kind in child_kinds:
            pytest.fail(f"child region {region.kind} should be subsumed by block")


def test_gated_delta_net_fusion_map_absorbs_module_ops() -> None:
    graph = _gdn_graph()
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    region_op_id = lowered.nodes[0].id
    assert len(lowered.fusion_map) == len(graph.node_order)
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gated_delta_net_decode_variant_flops() -> None:
    cfg = _mini_cfg()
    graph = _gdn_graph(seq_len=1, cfg=cfg, with_states=True)
    ctx = _fused_context(phase="decode")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/gated_delta_net/decode"
    expected = DEFAULT_GATED_DELTA_NET_RECIPE.decode_forward_flops(cfg)
    assert lowered.nodes[0].forward_flops == expected


def test_gated_delta_net_persist_conv_and_scan_ports() -> None:
    graph = _gdn_graph()
    lowered = lower(graph, _fused_context())
    persists = [
        ev for ev in lowered.nodes[0].resource_events if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) >= 2


def test_gated_delta_net_fla_layer_parity_excludes_l2_when_flagged() -> None:
    cfg = _mini_cfg()
    seq_len = 4
    default = DEFAULT_GATED_DELTA_NET_RECIPE.forward_flops(cfg, seq_len=seq_len)
    fla = FLA_LAYER_PARITY_GATED_DELTA_NET_RECIPE.forward_flops(cfg, seq_len=seq_len)
    assert fla < default
    assert fla == default - 6 * seq_len * cfg.num_qk_heads * cfg.head_dim
    graph = _gdn_graph(cfg=cfg)
    ctx = _fused_context(
        hardware="cuda",
        region_implementation_pins=MappingProxyType(
            {"region/gated_delta_net": "region/gated_delta_net/fla_layer_parity"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].forward_flops == fla


def test_gated_delta_net_pin_composed_tier_a() -> None:
    graph = _gdn_graph()
    ctx = _fused_context(
        module_implementation_pins=MappingProxyType(
            {"GatedDeltaNet": "region/gated_delta_net/composed_tier_a"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/gated_delta_net/composed_tier_a"


@pytest.mark.skip(reason="Atto golden block table not pinned in derivations yet")
def test_gated_delta_net_golden_vs_atto_block() -> None:
    cfg = GatedDeltaNetConfig(
        hidden_size=5120,
        num_qk_heads=16,
        num_v_heads=48,
        head_dim=128,
    )
    seq_len = 8192
    _ = DEFAULT_GATED_DELTA_NET_RECIPE.forward_flops(cfg, seq_len=seq_len)


def test_gated_delta_net_horizon_still_runs() -> None:
    cfg = GatedDeltaNetConfig(
        hidden_size=128,
        num_qk_heads=2,
        num_v_heads=2,
        head_dim=16,
    )
    conv_c = 2 * cfg.qk_proj_size + cfg.v_proj_size
    ctx = reference_invocation(
        default_dtype=DType.BF16,
        requested_capabilities=frozenset({"fused"}),
    )
    spec = HorizonSpec.inference(
        prefill=32,
        decode_steps=4,
        conv=ConvStateConfig(
            layer_indices=(0,),
            channels=conv_c,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=((0, "gated_delta", 2, 16, 16),),
            dtype=DType.BF16,
        ),
    )

    def module_fn(_compose):
        return GatedDeltaNet(cfg)

    def inputs_fn(step, _ctx, _state):
        return (Tensor(shape=(step.seq_len, cfg.hidden_size)),)

    report = estimate_horizon(spec, module_fn, inputs_fn, ctx)
    assert len(report.per_step) == 5
