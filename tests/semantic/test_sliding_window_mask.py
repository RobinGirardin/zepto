"""Tests for MaterializedSlidingWindowCausalMask operation."""

from __future__ import annotations

import pytest

from zepto.analysis import lower, reference_invocation
from zepto.compose import compose_graph
from zepto.modules.sliding_window_causal_mask import SlidingWindowCausalMask
from zepto.semantic import MaterializedSlidingWindowCausalMask, ResourceEventKind


def test_infer_outputs_shape() -> None:
    op = MaterializedSlidingWindowCausalMask(seq_len=16, window_size=8)
    (mask,) = op.infer_outputs(())
    assert mask.shape == (1, 16, 16)
    assert mask.persistent is True


def test_validation_positive_dims() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        MaterializedSlidingWindowCausalMask(seq_len=0, window_size=4).infer_outputs(())
    with pytest.raises(ValueError, match="window_size"):
        MaterializedSlidingWindowCausalMask(seq_len=8, window_size=0).infer_outputs(())
    with pytest.raises(ValueError, match="exceed"):
        MaterializedSlidingWindowCausalMask(seq_len=4, window_size=8).infer_outputs(())


def test_compose_module_persistent_alloc() -> None:
    graph = compose_graph(
        lambda _ctx: SlidingWindowCausalMask(16, 8),
        (),
    )
    lowered = lower(graph, reference_invocation())
    events = [
        ev
        for node in lowered.nodes
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    ]
    persist = [
        ev
        for node in lowered.nodes
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(events) >= 1
    assert len(persist) >= 1
