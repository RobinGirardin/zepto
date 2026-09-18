"""Tests for AttentionSoftmaxWithSink operation."""

from __future__ import annotations

import pytest

from zepto.compose import Parameter, Tensor, compose_graph
from zepto.modules.attention.attention_softmax_with_sink import AttentionSoftmaxWithSink
from zepto.semantic import AttentionSoftmaxWithSink as AttentionSoftmaxWithSinkOp


def test_infer_outputs_preserves_shape() -> None:
    scores = Tensor(shape=(8, 16, 16), requires_grad=False)
    sink = Tensor(shape=(8,), semantic_type="attention_sink", requires_grad=False)
    op = AttentionSoftmaxWithSinkOp()
    (weights,) = op.infer_outputs((scores,), (sink,))
    assert weights.shape == (8, 16, 16)


def test_forward_flops_formula() -> None:
    from zepto.semantic import EstimationContext
    from zepto.analysis.resolved import ResolvedValue
    from zepto.semantic.metadata import DType, TensorRole

    output = Tensor(shape=(4, 8, 8), requires_grad=False)
    ctx = EstimationContext(
        port_values=(
            ("output", ResolvedValue(tensor=output, role=TensorRole.ACTIVATION, dtype=DType.FP32)),
        )
    )
    op = AttentionSoftmaxWithSinkOp()
    assert op.forward_flops(ctx) == 5 * 4 * 8 * 8 + 4 * 8


def test_attention_softmax_with_sink_backward_flops() -> None:
    from zepto.semantic import EstimationContext
    from zepto.analysis.resolved import ResolvedValue
    from zepto.semantic.metadata import DType, TensorRole

    scores = Tensor(shape=(4, 8, 8), requires_grad=True)
    output = Tensor(shape=(4, 8, 8), requires_grad=False)
    ctx = EstimationContext(
        port_values=(
            ("scores", ResolvedValue(tensor=scores, role=TensorRole.ACTIVATION, dtype=DType.FP32)),
            ("output", ResolvedValue(tensor=output, role=TensorRole.ACTIVATION, dtype=DType.FP32)),
        )
    )
    op = AttentionSoftmaxWithSinkOp()
    assert op.backward_flops(ctx) == 4 * 4 * 8 * (8 + 1)
    assert op.saved_for_backward((scores,), (), (output,)) == ("output",)
    assert op.backward.supported is True


def test_compose_module_uses_sink_family() -> None:
    graph = compose_graph(
        lambda _ctx: AttentionSoftmaxWithSink(4),
        (Tensor(shape=(4, 8, 8), requires_grad=False),),
    )
    families = [graph.node(n).operation_family for n in graph.nodes]
    assert "attention_softmax_with_sink" in families
    assert "exp" not in families
    assert "reduce_sum" not in families
