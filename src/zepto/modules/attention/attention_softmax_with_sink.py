"""Attention softmax module with per-head sink parameter."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import AttentionSoftmaxWithSink as AttentionSoftmaxWithSinkOp


class AttentionSoftmaxWithSink(Module):
    """Scaled softmax over attention scores with learned per-head sink."""

    module_kind = "AttentionSoftmaxWithSink"

    def __init__(self, num_q_heads: int, *, scale: float | None = None) -> None:
        super().__init__()
        if num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        self.num_q_heads = num_q_heads
        self.scale = scale
        self.sink = Parameter(
            shape=(num_q_heads,),
            semantic_type="attention_sink",
        )

    def forward(self, scores: Tensor) -> Tensor:
        return AttentionSoftmaxWithSinkOp(scale=self.scale)(  # type: ignore[return-value]
            scores,
            parameters=(self.sink,),
        )


__all__ = ["AttentionSoftmaxWithSink"]
