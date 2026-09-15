"""Tests for CUDA cuBLAS workspace runtime overhead policy."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import (
    NullRuntimeOverheadPolicy,
    account_memory,
    estimate,
    reference_invocation,
    runtime_workspace_bytes,
)
from zepto.analysis.runtime import CudaCublasWorkspacePolicy
from zepto.compose import compose_graph
from zepto.compose.values import Tensor as ComposeTensor
from zepto.modules.linear import Linear

from test_memory_simulator import (
    _merge_train_lowered,
    build_matmul_relu_matmul_graph,
)

PRE_SM90 = 8_519_680
SM90_PLUS = 33_554_432


def test_missing_capability_workspace_zero() -> None:
    ctx = reference_invocation(hardware="cuda")
    assert runtime_workspace_bytes(ctx) == 0


def test_pre_sm90_workspace_forward() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
        phase="forward",
    )
    assert runtime_workspace_bytes(ctx) == PRE_SM90


def test_sm90_plus_workspace_forward() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(9, 0),
        phase="forward",
    )
    assert runtime_workspace_bytes(ctx) == SM90_PLUS


def test_train_phase_doubles_workspace() -> None:
    graph = build_matmul_relu_matmul_graph(batch=4, seq=8, hidden=32, inner=16)
    lowered = _merge_train_lowered(
        graph,
        hardware="cuda",
        compute_capability=(8, 0),
    )
    report = account_memory(lowered)
    assert report.breakdown.runtime_workspace == 2 * PRE_SM90


def test_backward_step_two_handles() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
        phase="backward",
    )
    assert runtime_workspace_bytes(ctx) == 2 * PRE_SM90


def test_non_cuda_zero() -> None:
    ctx = reference_invocation(
        hardware="generic",
        compute_capability=(8, 0),
    )
    assert runtime_workspace_bytes(ctx) == 0


def test_peak_includes_runtime() -> None:
    graph = compose_graph(
        lambda _ctx: Linear(32, 16),
        (ComposeTensor(shape=(2, 8, 32)),),
    )
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
    )
    report = estimate(graph, ctx)
    assert report.memory.peak_live_bytes >= report.memory.breakdown.runtime_workspace
    assert report.memory.breakdown.runtime_workspace == PRE_SM90


def test_graph_workspace_unchanged() -> None:
    graph = build_matmul_relu_matmul_graph(batch=4, seq=8, hidden=32, inner=16)
    lowered_base = _merge_train_lowered(graph, hardware="cuda")
    lowered_cuda = _merge_train_lowered(
        graph,
        hardware="cuda",
        compute_capability=(8, 0),
    )
    base_report = account_memory(lowered_base)
    cuda_report = account_memory(lowered_cuda)
    assert cuda_report.breakdown.workspace == base_report.breakdown.workspace
    assert cuda_report.breakdown.runtime_workspace == 2 * PRE_SM90
    assert cuda_report.breakdown.workspace >= 0


def test_null_runtime_policy() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
        runtime_policy=NullRuntimeOverheadPolicy(),
    )
    assert runtime_workspace_bytes(ctx) == 0


def test_policy_on_context_override() -> None:
    ctx = reference_invocation(hardware="cuda")
    ctx = replace(
        ctx,
        compute_capability=(8, 0),
        runtime_policy=CudaCublasWorkspacePolicy(compute_capability=(8, 0)),
    )
    assert runtime_workspace_bytes(ctx) == PRE_SM90
