"""RoPE materialize/apply region discovery and FLOP recipes."""

from __future__ import annotations

from zepto.analysis import account_flops, discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.rope import (
    DEFAULT_ROPE_APPLY_RECIPE,
    DEFAULT_ROPE_MATERIALIZE_RECIPE,
)
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.attention.gqa import GroupedQueryAttention
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_materialize import RoPEMaterialize

_SEQ = 8
_HEADS = 4
_HEAD_DIM = 8


def _registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_defaults(registry)
    return registry


def test_rope_materialize_region_flops() -> None:
    graph = compose_graph(
        lambda _ctx: RoPEMaterialize(_SEQ, _HEAD_DIM),
        (),
    )
    ctx = reference_invocation()
    registry = _registry()
    regions = [
        r
        for r in discover_regions(graph, ctx, registry)
        if r.kind == "region/rope_materialize"
    ]
    assert len(regions) == 1
    lowered = lower(graph, ctx, registry=registry)
    mat_nodes = [
        n
        for n in lowered.nodes
        if n.implementation.startswith("region/rope_materialize")
    ]
    assert len(mat_nodes) == 1
    expected = DEFAULT_ROPE_MATERIALIZE_RECIPE.forward_flops(
        seq_len=_SEQ, head_dim=_HEAD_DIM
    )
    assert mat_nodes[0].forward_flops == expected
    assert mat_nodes[0].backward_flops == 0
    # Region absorbs inner ops — no duplicate matmul/cos/sin leaves.
    assert not any(
        n.implementation.endswith("/identity")
        and graph.node(n.node_id).operation_family in ("matmul", "cos", "sin")
        for n in lowered.nodes
        if n.node_id is not None
    )


def test_rope_apply_region_flops_per_tensor() -> None:
    def factory(_ctx):
        mat = RoPEMaterialize(_SEQ, _HEAD_DIM)
        apply = RoPEApply(_HEAD_DIM)

        class _Harness(Module):
            def __init__(self) -> None:
                super().__init__()
                self._q = Tensor(
                    shape=(_HEADS, _SEQ, _HEAD_DIM), requires_grad=True
                )

            def forward(self) -> Tensor:
                cos, sin = mat()
                return apply(self._q, cos, sin)

        return _Harness()

    graph = compose_graph(factory, ())
    ctx = reference_invocation()
    registry = _registry()
    lowered = lower(graph, ctx, registry=registry)
    apply_nodes = [
        n
        for n in lowered.nodes
        if n.implementation.startswith("region/rope_apply")
    ]
    assert len(apply_nodes) == 1
    expected = DEFAULT_ROPE_APPLY_RECIPE.forward_flops(
        num_heads=_HEADS, seq_len=_SEQ, head_dim=_HEAD_DIM
    )
    assert apply_nodes[0].forward_flops == expected
    assert apply_nodes[0].backward_flops == expected


def test_gqa_with_rope_apply_regions() -> None:
    graph = compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HEADS * _HEAD_DIM,
            _HEADS,
            2,
            head_dim=_HEAD_DIM,
            rope=RoPEApply(_HEAD_DIM),
        ),
        (
            Tensor(shape=(_SEQ, _HEADS * _HEAD_DIM), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
            Tensor(shape=(_SEQ, _HEAD_DIM), requires_grad=False),
            Tensor(shape=(_SEQ, _HEAD_DIM), requires_grad=False),
        ),
    )
    ctx = reference_invocation()
    registry = _registry()
    apply_regions = [
        r
        for r in discover_regions(graph, ctx, registry)
        if r.kind == "region/rope_apply"
    ]
    # One provenance region covers both Q and K apply invocations on the same module.
    assert len(apply_regions) == 1
    q_flops = DEFAULT_ROPE_APPLY_RECIPE.forward_flops(
        num_heads=_HEADS, seq_len=_SEQ, head_dim=_HEAD_DIM
    )
    k_flops = DEFAULT_ROPE_APPLY_RECIPE.forward_flops(
        num_heads=2, seq_len=_SEQ, head_dim=_HEAD_DIM
    )
    lowered = lower(graph, ctx, registry=registry)
    apply_flops = sum(
        n.forward_flops
        for n in lowered.nodes
        if n.implementation.startswith("region/rope_apply")
    )
    assert apply_flops == q_flops + k_flops
    report = account_flops(lowered)
    assert report.forward_flops > apply_flops
