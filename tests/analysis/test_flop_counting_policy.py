"""FLOP counting policy: dual Zepto / FlopCounterMode totals."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import (
    account_fcm_flops,
    account_flops,
    lower,
    reference_invocation,
)
from zepto.analysis.lowering.helpers import build_estimation_context
from zepto.compose import Module, Parameter, Tensor, compose_graph
from zepto.modules.layers.linear import Linear
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.semantic import Add, LinearMatMul

from test_flop_accounting import _maximum_graph

_IN_FEATURES = 8
_OUT_FEATURES = 16
_HIDDEN = 8


def _linear_graph():
    return compose_graph(
        lambda _ctx: Linear(_IN_FEATURES, _OUT_FEATURES),
        (Tensor(shape=(2, _IN_FEATURES), requires_grad=True),),
    )


class _LinearResidual(Module):
    """One LinearMatMul plus a same-shape residual Add."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = Parameter(shape=(_HIDDEN, _HIDDEN), semantic_type="weight")

    def forward(self, x: Tensor) -> Tensor:
        projected = LinearMatMul()(x, parameters=(self.weight,))
        return Add()(x, projected)  # type: ignore[return-value]


def _linear_residual_graph():
    return compose_graph(
        lambda _ctx: _LinearResidual(),
        (Tensor(shape=(2, _HIDDEN), requires_grad=True),),
    )


def _rmsnorm_graph():
    return compose_graph(
        lambda _ctx: RMSNorm(_HIDDEN),
        (Tensor(shape=(2, _HIDDEN), requires_grad=True),),
    )


def _declaration_flops(graph, context, family: str) -> tuple[int, int]:
    nodes = [
        graph.node(node_id)
        for node_id in graph.node_order
        if graph.node(node_id).operation_family == family
    ]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.declaration is not None
    estimation = build_estimation_context(node, graph, context)
    forward = node.declaration.forward_flops(estimation)
    backward = node.declaration.backward_flops(replace(estimation, phase="backward"))
    return forward, backward


def test_reference_invocation_default_flop_policy() -> None:
    assert reference_invocation().flop_policy == "zepto"


def test_account_flops_named_fields_on_maximum() -> None:
    graph = _maximum_graph()
    lowered = lower(graph, reference_invocation(phase="forward"))
    report = account_flops(lowered)

    forward = sum(node.forward_flops for node in lowered.nodes)
    backward = sum(node.backward_flops for node in lowered.nodes)

    assert lowered.source_graph is graph
    assert report.flop_policy == "zepto"
    assert report.zepto_forward_flops == forward
    assert report.zepto_backward_flops == backward
    assert report.zepto_total_flops == forward
    assert report.fcm_forward_flops == 0
    assert report.fcm_backward_flops == 0
    assert report.fcm_total_flops == 0
    assert report.forward_flops == report.zepto_forward_flops


def test_flop_counter_mode_policy_selects_zero_fcm_totals() -> None:
    graph = _maximum_graph()
    ctx = replace(reference_invocation(phase="forward"), flop_policy="flop_counter_mode")
    lowered = lower(graph, ctx)
    report = account_flops(lowered)

    forward = sum(node.forward_flops for node in lowered.nodes)
    backward = sum(node.backward_flops for node in lowered.nodes)

    assert report.flop_policy == "flop_counter_mode"
    assert report.total_flops == 0
    assert report.forward_flops == 0
    assert report.backward_flops == 0
    assert report.fcm_forward_flops == 0
    assert report.fcm_backward_flops == 0
    assert report.fcm_total_flops == 0
    assert report.zepto_forward_flops == forward
    assert report.zepto_backward_flops == backward
    assert report.zepto_total_flops == forward


def test_fcm_matches_zepto_on_linear_only() -> None:
    graph = _linear_graph()
    ctx = reference_invocation(phase="forward")
    lowered = lower(graph, ctx)
    report = account_flops(lowered)
    expected_fwd, expected_bwd = _declaration_flops(graph, ctx, "linear_matmul")
    fcm_fwd, fcm_bwd = account_fcm_flops(graph, ctx)

    assert report.fcm_forward_flops == report.zepto_forward_flops == expected_fwd
    assert report.fcm_backward_flops == expected_bwd
    assert report.fcm_forward_flops == fcm_fwd
    assert report.fcm_backward_flops == fcm_bwd
    assert report.forward_flops == report.zepto_forward_flops


def test_fcm_drops_residual_add() -> None:
    graph = _linear_residual_graph()
    ctx = reference_invocation(phase="forward")
    lowered = lower(graph, ctx)
    report = account_flops(lowered)
    linear_fwd, linear_bwd = _declaration_flops(graph, ctx, "linear_matmul")
    families = {graph.node(node_id).operation_family for node_id in graph.node_order}

    assert "linear_matmul" in families
    assert "add" in families
    assert report.zepto_forward_flops > report.fcm_forward_flops
    assert report.fcm_forward_flops == linear_fwd
    assert report.fcm_backward_flops == linear_bwd
    assert report.fcm_forward_flops == account_fcm_flops(graph, ctx)[0]


def test_fcm_zero_on_rmsnorm_only() -> None:
    graph = _rmsnorm_graph()
    ctx = reference_invocation(phase="forward")
    lowered = lower(graph, ctx)
    report = account_flops(lowered)

    assert report.zepto_forward_flops > 0
    assert report.fcm_forward_flops == 0
    assert report.fcm_backward_flops == 0
    assert report.fcm_total_flops == 0
    assert account_fcm_flops(graph, ctx) == (0, 0)


def test_flop_counter_mode_policy_selects_fcm_on_linear_residual() -> None:
    graph = _linear_residual_graph()
    ctx = replace(
        reference_invocation(phase="forward"),
        flop_policy="flop_counter_mode",
    )
    lowered = lower(graph, ctx)
    report = account_flops(lowered)

    assert report.flop_policy == "flop_counter_mode"
    assert report.forward_flops == report.fcm_forward_flops
    assert report.backward_flops == report.fcm_backward_flops
    assert report.total_flops == report.fcm_total_flops
    assert report.zepto_forward_flops > report.fcm_forward_flops
