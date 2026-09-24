"""Tests for EmbeddingLookup batched token indices."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor
from zepto.semantic import EmbeddingLookup


def test_infer_outputs_rank1() -> None:
    op = EmbeddingLookup()
    (output,) = op.infer_outputs(
        (Tensor(shape=(12,), semantic_type="token_ids"),),
        parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
    )
    assert output.shape == (12, 64)


def test_infer_outputs_rank2_batched() -> None:
    op = EmbeddingLookup()
    (output,) = op.infer_outputs(
        (Tensor(shape=(4, 12), semantic_type="token_ids"),),
        parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
    )
    assert output.shape == (4, 12, 64)


def test_invalid_token_rank_raises() -> None:
    op = EmbeddingLookup()
    with pytest.raises(ValueError, match="rank-1 \\(S,\\) or rank-2"):
        op.infer_outputs(
            (Tensor(shape=(2, 4, 8), semantic_type="token_ids"),),
            parameters=(Tensor(shape=(64, 32000), semantic_type="weight"),),
        )


def test_weight_grad_persists_on_full_phase() -> None:
    from zepto.analysis import lower, reference_invocation
    from zepto.analysis.memory.simulator import ResourceEventSimulator
    from zepto.compose import compose_graph
    from zepto.modules.layers.embedding import Embedding
    from zepto.semantic import ResourceEventKind

    hidden, vocab, seq_len = 8, 16, 4
    graph = compose_graph(
        lambda _ctx: Embedding(hidden, vocab),
        (Tensor(shape=(seq_len,), semantic_type="token_ids"),),
    )
    lowered = lower(graph, reference_invocation(phase="full"))
    persist = [
        event
        for node in lowered.nodes
        for event in node.resource_events
        if event.kind is ResourceEventKind.PERSIST
    ]
    assert persist
    result = ResourceEventSimulator(lowered).run()
    assert result.breakdown.weight_grads == hidden * vocab * 4
