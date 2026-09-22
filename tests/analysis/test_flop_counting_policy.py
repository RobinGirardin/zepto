"""FLOP counting policy: dual Zepto / FlopCounterMode totals."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import (
    account_flops,
    lower,
    reference_invocation,
)

from test_flop_accounting import _maximum_graph


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
