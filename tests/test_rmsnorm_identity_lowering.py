"""Identity-lowering oracle for decomposed RMSNorm (9-op HF eager chain)."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.rms_norm import RMSNorm
from zepto.semantic import ResourceEventKind

S, D = 4, 8


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def _lower_rmsnorm_identity(
    *,
    shape: tuple[int, ...] = (S, D),
    requires_grad: bool = True,
):
    graph = compose_graph(
        lambda ctx: RMSNorm(shape[-1]),
        (Tensor(shape=shape, requires_grad=requires_grad),),
    )
    return lower(graph, reference_invocation(), registry=_identity_registry()), graph


def test_rmsnorm_identity_lowers_nine_ops() -> None:
    lowered, graph = _lower_rmsnorm_identity()
    assert len(graph.node_order) == 9
    assert len(lowered.nodes) == 9
    assert not lowered.fusion_map


def test_rmsnorm_identity_op_families_in_order() -> None:
    _, graph = _lower_rmsnorm_identity()
    families = [graph.node(op_id).operation_family for op_id in graph.node_order]
    assert families == [
        "cast",
        "multiply",
        "reduce_sum",
        "divide",
        "add",
        "square_root",
        "divide",
        "cast",
        "parameter_scale",
    ]


def test_rmsnorm_identity_forward_flops() -> None:
    """Primitive sum: 4·S·d + 2·S (casts bill 0 FLOPs)."""
    lowered, _ = _lower_rmsnorm_identity()
    numel = S * D
    expected = 4 * numel + 2 * S
    assert sum(op.forward_flops for op in lowered.nodes) == expected


def test_rmsnorm_identity_allocate_count() -> None:
    lowered, _ = _lower_rmsnorm_identity()
    alloc = sum(
        1
        for op in lowered.nodes
        for ev in op.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert alloc == 9


def test_rmsnorm_identity_save_events_training() -> None:
    lowered, _ = _lower_rmsnorm_identity(requires_grad=True)
    saves = [
        ev
        for op in lowered.nodes
        for ev in op.resource_events
        if ev.kind is ResourceEventKind.SAVE
    ]
    assert len(saves) >= 1
    unique = {ev.value for ev in saves}
    assert len(unique) >= 3
