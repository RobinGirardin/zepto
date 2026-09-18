"""Discovery, lowering, and variant tests for region/mamba2_mixer."""

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
    DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
    DEFAULT_MAMBA2_MIXER_RECIPE,
    DEFAULT_MAMBA2_SCAN_RECIPE,
    Mamba2MixerRecipe,
)
from zepto.compose import Tensor, compose_graph
from modules.mixers.mamba2_mixer import Mamba2Mixer
from modules.mixers.mixer_config import Mamba2MixerConfig
from zepto.semantic import ResourceEventKind
from zepto.semantic.metadata import DType


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _mini_cfg() -> Mamba2MixerConfig:
    return Mamba2MixerConfig(
        hidden_size=32,
        num_heads=2,
        head_dim=16,
        state_size=8,
        num_groups=2,
    )


def _mixer_graph(
    *,
    seq_len: int = 4,
    cfg: Mamba2MixerConfig | None = None,
    requires_grad: bool = True,
    with_states: bool = False,
):
    cfg = cfg or _mini_cfg()
    inputs: tuple[Tensor, ...] = (
        Tensor(shape=(seq_len, cfg.hidden_size), requires_grad=requires_grad),
    )
    if with_states:
        inputs = (
            *inputs,
            Tensor(shape=(cfg.conv_channels, cfg.conv_kernel_size), requires_grad=False),
            Tensor(
                shape=(cfg.num_heads, cfg.head_dim, cfg.state_size),
                requires_grad=False,
            ),
        )
    return compose_graph(lambda _ctx: Mamba2Mixer(cfg), inputs)


def _leaf_sum_flops(cfg: Mamba2MixerConfig, *, seq_len: int, recipe: Mamba2MixerRecipe):
    s = seq_len
    d, h, p, n = cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.state_size
    m = cfg.intermediate_size
    c_c = cfg.conv_channels
    p_in = cfg.in_proj_size
    conv_recipe = recipe._conv_recipe()
    linears = 2 * s * d * p_in + 2 * s * m * d
    conv = conv_recipe.forward_flops(s * c_c)
    scan = DEFAULT_MAMBA2_SCAN_RECIPE.forward_flops(
        seq_len=s, num_heads=h, head_dim=p, state_size=n
    )
    norm = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE.forward_flops(s * m)
    return linears + conv + scan + norm


def test_compose_mamba2_mixer_discovers_block_region() -> None:
    graph = _mixer_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    blocks = [r for r in regions if r.kind == "region/mamba2_mixer"]
    assert len(blocks) == 1
    assert blocks[0].anchor.component_type == "Mamba2Mixer"
    assert blocks[0].matcher_id == "prov-mamba2-mixer"
    assert len(blocks[0].operation_ids) == len(graph.node_order)


def test_mamba2_mixer_unfused_without_fused_capability() -> None:
    graph = _mixer_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) > 1


def test_mamba2_mixer_fused_lowering_flops_and_allocs() -> None:
    cfg = _mini_cfg()
    recipe = DEFAULT_MAMBA2_MIXER_RECIPE
    seq_len = 4
    assert recipe.forward_flops(cfg, seq_len=seq_len) == _leaf_sum_flops(
        cfg, seq_len=seq_len, recipe=recipe
    )
    graph = _mixer_graph(seq_len=seq_len, cfg=cfg)
    lowered = lower(graph, _fused_context())
    assert lowered.nodes[0].forward_flops == recipe.forward_flops(cfg, seq_len=seq_len)
    saves = [
        ev for ev in lowered.nodes[0].resource_events if ev.kind is ResourceEventKind.SAVE
    ]
    assert len(saves) == 3
    allocates = [
        ev
        for ev in lowered.nodes[0].resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    ]
    assert len(allocates) == 4


def test_mamba2_mixer_fusion_map_absorbs_all_ops() -> None:
    graph = _mixer_graph()
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    region_op_id = lowered.nodes[0].id
    assert len(lowered.fusion_map) == len(graph.node_order)
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_mamba2_mixer_overlap_subsumes_child_regions() -> None:
    graph = _mixer_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context()
    regions = discover_regions(graph, ctx, registry)
    winners = resolve_overlaps(graph, regions, ctx, registry)
    block_winners = [r for r in winners if r.kind == "region/mamba2_mixer"]
    assert len(block_winners) == 1
    child_kinds = {
        "region/mamba2_scan",
        "region/gated_grouped_rms_norm",
        "region/depthwise_causal_conv1d",
    }
    for region in winners:
        if region.kind in child_kinds:
            pytest.fail(f"child region {region.kind} should be subsumed by block")


def test_mamba2_mixer_decode_variant_with_state_ports() -> None:
    cfg = _mini_cfg()
    graph = _mixer_graph(seq_len=1, cfg=cfg, with_states=True)
    ctx = _fused_context(phase="decode")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/mamba2_mixer/decode"
    expected = DEFAULT_MAMBA2_MIXER_RECIPE.decode_forward_flops(cfg)
    assert lowered.nodes[0].forward_flops == expected


def test_mamba2_mixer_hub_mega_wins_on_cuda() -> None:
    graph = _mixer_graph()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/mamba2_mixer/hub_mega"


def test_mamba2_mixer_golden_nemotron_mini_block_flops() -> None:
    cfg = Mamba2MixerConfig(
        hidden_size=2688,
        num_heads=64,
        head_dim=64,
        state_size=128,
        num_groups=8,
    )
    seq_len = 8192
    recipe = DEFAULT_MAMBA2_MIXER_RECIPE
    conv_recipe = recipe._conv_recipe()
    s, d, h, p, n = seq_len, cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.state_size
    m = cfg.intermediate_size
    c_c = cfg.conv_channels
    p_in = cfg.in_proj_size
    expected = (
        2 * s * d * p_in
        + 2 * s * m * d
        + conv_recipe.forward_flops(s * c_c)
        + DEFAULT_MAMBA2_SCAN_RECIPE.forward_flops(
            seq_len=s, num_heads=h, head_dim=p, state_size=n
        )
        + DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE.forward_flops(s * m)
    )
    assert recipe.forward_flops(cfg, seq_len=seq_len) == expected


def test_mamba2_mixer_horizon_still_runs() -> None:
    cfg = _mini_cfg()
    ctx = reference_invocation(
        default_dtype=DType.BF16,
        requested_capabilities=frozenset({"fused"}),
    )
    spec = HorizonSpec.inference(
        prefill=32,
        decode_steps=4,
        conv=ConvStateConfig(
            layer_indices=(0,),
            channels=cfg.conv_channels,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=((0, "mamba2", cfg.num_heads, cfg.head_dim, cfg.state_size),),
            dtype=DType.BF16,
        ),
    )

    def module_fn(_compose):
        return Mamba2Mixer(cfg)

    def inputs_fn(step, _ctx, _state):
        return (Tensor(shape=(step.seq_len, cfg.hidden_size)),)

    report = estimate_horizon(spec, module_fn, inputs_fn, ctx)
    assert len(report.per_step) == 5


def test_mamba2_mixer_pin_composed_tier_a() -> None:
    graph = _mixer_graph()
    ctx = _fused_context(
        hardware="cuda",
        module_implementation_pins=MappingProxyType(
            {"Mamba2Mixer": "region/mamba2_mixer/composed_tier_a"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/mamba2_mixer/composed_tier_a"
