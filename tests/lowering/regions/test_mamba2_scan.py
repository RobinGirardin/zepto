"""Discovery, lowering, and variant tests for region/mamba2_scan."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.horizon.state import RecurrentScanState
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.mixers.selective_ssm_scan import SelectiveSSMScan
from zepto.semantic import ResourceEventKind
from zepto.semantic.metadata import DType


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _scan_graph(
    *,
    seq_len: int = 4,
    num_heads: int = 4,
    head_dim: int = 8,
    state_size: int = 16,
    num_groups: int = 2,
    requires_grad: bool = True,
    with_scan_state_in: bool = False,
):
    inputs: tuple[Tensor, ...] = (
        Tensor(shape=(seq_len, num_heads, head_dim), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_heads), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_groups, state_size), requires_grad=requires_grad),
        Tensor(shape=(seq_len, num_groups, state_size), requires_grad=requires_grad),
    )
    if with_scan_state_in:
        inputs = (
            *inputs,
            Tensor(
                shape=(num_heads, head_dim, state_size),
                requires_grad=False,
            ),
        )

    return compose_graph(
        lambda _ctx: SelectiveSSMScan(
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            num_groups=num_groups,
        ),
        inputs,
    )


def _forward_flops(
    *, seq_len: int, num_heads: int, head_dim: int, state_size: int
) -> int:
    s, h, p, n = seq_len, num_heads, head_dim, state_size
    return s * h * (5 * p * n + n + 2 * p + 7)


def _backward_flops(
    *, seq_len: int, num_heads: int, head_dim: int, state_size: int
) -> int:
    s, h, p, n = seq_len, num_heads, head_dim, state_size
    return s * h * (10 * p * n + 2 * p + n + 4)


def test_compose_mamba2_scan_discovers_region() -> None:
    graph = _scan_graph(seq_len=3)
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    scan = [r for r in regions if r.kind == "region/mamba2_scan"]
    assert len(scan) == 1
    assert scan[0].anchor.component_type == "SelectiveSSMScan"
    assert scan[0].matcher_id == "prov-mamba2-scan"


def test_mamba2_scan_unfused_without_capability() -> None:
    graph = _scan_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) > 1


def test_mamba2_scan_fused_lowering_flops_and_allocs() -> None:
    seq_len, h, p, n = 4, 4, 8, 16
    graph = _scan_graph(
        seq_len=seq_len, num_heads=h, head_dim=p, state_size=n
    )
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/mamba2_scan/reference"
    assert op.forward_flops == _forward_flops(
        seq_len=seq_len, num_heads=h, head_dim=p, state_size=n
    )
    assert op.backward_flops == _backward_flops(
        seq_len=seq_len, num_heads=h, head_dim=p, state_size=n
    )
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("state_checkpoint" in aux for aux in op.auxiliary_edges)
    elided_aux = [
        aux
        for aux in op.auxiliary_edges
        if any(
            tag in aux
            for tag in (
                "a_bar",
                "b_bar",
                "decayed",
                "update",
                "weighted",
                "softplus",
            )
        )
    ]
    assert not elided_aux


def test_mamba2_scan_fusion_map_absorbs_all_ops() -> None:
    graph = _scan_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    n_ops = len(regions[0].operation_ids)
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == n_ops
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_mamba2_scan_ssd_chunk_wins_on_cuda() -> None:
    graph = _scan_graph()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/mamba2_scan"


def test_mamba2_scan_decode_variant_wins_on_cuda() -> None:
    h, p, n = 4, 8, 16
    graph = _scan_graph(
        seq_len=1,
        num_heads=h,
        head_dim=p,
        state_size=n,
        with_scan_state_in=True,
    )
    scan_state = RecurrentScanState(
        layer_index=0,
        kind="mamba2",
        num_heads=h,
        head_dim=p,
        state_dim=n,
        dtype=DType.FP32,
    )
    ctx = _fused_context(
        hardware="cuda",
        state=(("scan_state:0", scan_state),),
    )
    lowered = lower(graph, ctx)
    op = lowered.nodes[0]
    assert op.implementation == "region/mamba2_scan/decode"
    assert op.forward_flops == _forward_flops(
        seq_len=1, num_heads=h, head_dim=p, state_size=n
    )
    persists = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.PERSIST]
    assert persists
    assert any("scan_state" in aux for aux in op.auxiliary_edges)
    assert len(lowered.state_port_events) == 1


def test_mamba2_scan_prefill_rejects_decode_graph_on_ssd_variant() -> None:
    h, p, n = 4, 8, 16
    graph = _scan_graph(
        seq_len=1,
        num_heads=h,
        head_dim=p,
        state_size=n,
        with_scan_state_in=True,
    )
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/mamba2_scan/decode"


def test_mamba2_scan_batched_flops_and_checkpoint_shape() -> None:
    from tests.lowering.regions._batch_helpers import assert_flops_scales, aux_tensor

    batch, seq_len, h, p, n, g = 2, 4, 4, 8, 16, 2
    single = lower(_scan_graph(seq_len=seq_len, num_heads=h, head_dim=p, state_size=n, num_groups=g), _fused_context())
    batched = lower(
        compose_graph(
            lambda _ctx: SelectiveSSMScan(
                num_heads=h, head_dim=p, state_size=n, num_groups=g
            ),
            (
                Tensor(shape=(batch, seq_len, h, p), requires_grad=True),
                Tensor(shape=(batch, seq_len, h), requires_grad=True),
                Tensor(shape=(batch, seq_len, g, n), requires_grad=True),
                Tensor(shape=(batch, seq_len, g, n), requires_grad=True),
            ),
        ),
        _fused_context(),
    )
    assert_flops_scales(
        batched.nodes[0].forward_flops, single.nodes[0].forward_flops, batch
    )
    assert aux_tensor(batched, batched.nodes[0], "state_checkpoint").shape == (
        batch,
        seq_len,
        h,
        p,
        n,
    )
