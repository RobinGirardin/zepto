"""Discovery, lowering, and variant tests for region/gated_delta_scan."""

from __future__ import annotations

from zepto.analysis import (
    ConvStateConfig,
    HorizonSpec,
    RecurrentStateConfig,
    discover_regions,
    estimate_horizon,
    lower,
    reference_invocation,
)
from zepto.analysis.horizon.state import RecurrentScanState
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from modules.mixers.gated_delta_net import GatedDeltaNet
from modules.mixers.gated_delta_scan import GatedDeltaScan
from modules.mixers.mixer_config import GatedDeltaNetConfig
from zepto.semantic import ResourceEventKind
from zepto.semantic.metadata import DType


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _scan_graph(
    *,
    seq_len: int = 4,
    num_heads: int = 2,
    key_dim: int = 8,
    value_dim: int = 8,
    requires_grad: bool = True,
    with_scan_state_in: bool = False,
):
    inputs: tuple[Tensor, ...] = (
        Tensor(shape=(seq_len, num_heads, key_dim), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_heads, key_dim), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_heads, value_dim), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_heads), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_heads), requires_grad=requires_grad),
    )
    if with_scan_state_in:
        inputs = (
            *inputs,
            Tensor(
                shape=(num_heads, key_dim, value_dim),
                requires_grad=False,
            ),
        )

    return compose_graph(
        lambda _ctx: GatedDeltaScan(
            num_heads=num_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
        ),
        inputs,
    )


def _forward_flops(
    *, seq_len: int, num_heads: int, key_dim: int, value_dim: int
) -> int:
    return seq_len * num_heads * (8 * key_dim * value_dim + 2 * value_dim + 1)


def _backward_flops(
    *, seq_len: int, num_heads: int, key_dim: int, value_dim: int
) -> int:
    return seq_len * num_heads * (16 * key_dim * value_dim + 4 * value_dim + 4)


def test_compose_gated_delta_scan_discovers_region() -> None:
    graph = _scan_graph(seq_len=3)
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    scan = [r for r in regions if r.kind == "region/gated_delta_scan"]
    assert len(scan) == 1
    assert scan[0].anchor.component_type == "GatedDeltaScan"
    assert scan[0].matcher_id == "prov-gated-delta-scan"


def test_gated_delta_scan_unfused_without_capability() -> None:
    graph = _scan_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) > 1


def test_gated_delta_scan_fused_lowering_flops_and_allocs() -> None:
    seq_len, h, dk, dv = 4, 2, 8, 8
    graph = _scan_graph(seq_len=seq_len, num_heads=h, key_dim=dk, value_dim=dv)
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/gated_delta_scan/reference"
    assert op.forward_flops == _forward_flops(
        seq_len=seq_len, num_heads=h, key_dim=dk, value_dim=dv
    )
    assert op.backward_flops == _backward_flops(
        seq_len=seq_len, num_heads=h, key_dim=dk, value_dim=dv
    )
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("state_checkpoint" in aux for aux in op.auxiliary_edges)
    elided_aux = [
        aux
        for aux in op.auxiliary_edges
        if any(
            aux.endswith(f"_{tag}") or f":{tag}_" in aux
            for tag in ("decayed", "prediction", "delta", "outer")
        )
    ]
    assert not elided_aux


def test_gated_delta_scan_fusion_map_absorbs_all_ops() -> None:
    graph = _scan_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    n_ops = len(regions[0].operation_ids)
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == n_ops
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gated_delta_scan_fla_chunk_wins_on_cuda() -> None:
    graph = _scan_graph()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/gated_delta_scan"


def test_gated_delta_scan_decode_recurrent_variant() -> None:
    h, dk, dv = 2, 8, 8
    graph = _scan_graph(
        seq_len=1, num_heads=h, key_dim=dk, value_dim=dv, with_scan_state_in=True
    )
    scan_state = RecurrentScanState(
        layer_index=0,
        kind="gated_delta",
        num_heads=h,
        head_dim=dk,
        state_dim=dv,
        dtype=DType.FP32,
    )
    ctx = _fused_context(
        hardware="cuda",
        state=(("scan_state:0", scan_state),),
    )
    lowered = lower(graph, ctx)
    op = lowered.nodes[0]
    assert op.implementation == "region/gated_delta_scan/decode"
    assert op.forward_flops == _forward_flops(
        seq_len=1, num_heads=h, key_dim=dk, value_dim=dv
    )
    persists = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.PERSIST]
    assert persists
    assert any("scan_state" in aux for aux in op.auxiliary_edges)
    assert len(lowered.state_port_events) == 1


def test_gated_delta_scan_horizon_integration() -> None:
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
        hardware="cuda",
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
    assert report.state_final.custom
    assert report.per_step[1].flops.forward_flops < report.per_step[0].flops.forward_flops
