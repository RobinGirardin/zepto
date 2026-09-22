"""Attention softmax with per-query-head sink denominator term."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _batched_attention_dims(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return ``(batch, heads, seq_len)`` for rank-3 or rank-4 attention scores."""
    rank = len(shape)
    if rank == 3:
        heads, seq_len, _ = shape
        return 1, heads, seq_len
    if rank == 4:
        batch, heads, seq_len, _ = shape
        return batch, heads, seq_len
    raise ValueError(
        "attention_softmax_with_sink expects rank-3 (h, S, S) or rank-4 (B, h, S, S) "
        f"scores, got shape {shape}"
    )


@dataclass(frozen=True, slots=True)
class AttentionSoftmaxWithSink(Operation):
    """Stable softmax over attention scores with a learned sink per query head.

    GPT-OSS adds a per-head sink scalar to the softmax denominator (equivalent
    to an extra key slot absorbing probability mass). For structural estimation
    this is modeled as one fused operation rather than decomposed ``exp`` /
    ``reduce_sum`` / ``divide``.

    Forward FLOPs are approximated as standard stable softmax ``5 h S²`` plus
    ``h S`` for the sink contribution.
    """

    scale: float | None = None

    @property
    def family(self) -> str:
        return "attention_softmax_with_sink"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("scores"),)

    @property
    def parameter_ports(self) -> tuple[Port, ...]:
        return (
            Port(
                "sink",
                value_kind=ValueKind.PARAMETER,
                contract=None,
            ),
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=True, saved_for_backward=("output",))

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (scores,) = inputs
        _batched_attention_dims(scores.shape)
        if len(parameters) != 1:
            raise ValueError("attention_softmax_with_sink requires a sink parameter")
        return (scores,)

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        del parameters, outputs
        if inputs[0].requires_grad:
            return ("output",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        batch, heads, seq_len = _batched_attention_dims(output.shape)
        standard = 5 * batch * heads * seq_len * seq_len
        sink = batch * heads * seq_len
        return standard + sink

    def backward_flops(self, context: EstimationContext) -> int:
        from zepto.analysis.lowering.recipes.softmax_one import (
            MEGATRON_SOFTMAX_ONE_RECIPE,
        )

        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        batch, heads, seq_len = _batched_attention_dims(output.shape)
        scores = context.tensor_for("scores")
        requires_grad = scores.requires_grad if scores is not None else False
        bwd = MEGATRON_SOFTMAX_ONE_RECIPE.backward_flops(
            num_heads=int(heads),
            seq_len=int(seq_len),
            requires_grad=requires_grad,
        )
        return bwd * batch

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return allocate(len(self.output_ports))


__all__ = ["AttentionSoftmaxWithSink"]
