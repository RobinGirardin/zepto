"""Batched matrix multiplication of two data-flow tensors.

For activation × registered weight, use ``LinearMatMul`` instead.
"""

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_matmul_auxiliary_ports,
    allocate,
    broadcast_tensor,
    emit_binary_backward_resource_events,
    numel,
    reduced_gradient_tensor,
    unreduced_gradient_tensor_matmul,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _batch_tensor(value: Tensor) -> Tensor:
    """Return a tensor describing the leading batch prefix only."""
    return Tensor(shape=value.shape[:-2])


def _broadcast_batch(left: Tensor, right: Tensor, *, family: str) -> tuple[int, ...]:
    """Broadcast the leading batch dimensions of two matmul operands."""
    return broadcast_tensor(
        (_batch_tensor(left), _batch_tensor(right)),
        family=family,
    ).shape


def _batch_reduction_flops(operand: Tensor, output: Tensor) -> int:
    """Return FLOPs to sum a batched operand gradient over broadcast batch axes."""
    output_batch = numel(_batch_tensor(output))
    unreduced = output_batch * operand.shape[-2] * operand.shape[-1]
    return unreduced - numel(operand)


class MatMul(Operation):
    """Multiply two graph tensors along their final two dimensions.

    Both operands are data-flow tensors (``input_ports`` only). Model weights
    bound through parameter ports use the separate ``linear_matmul`` family.

    Leading dimensions broadcast NumPy-style between operands.
    """

    @property
    def family(self) -> str:
        """Return the stable matrix multiplication family name."""
        return "matmul"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        """Return the left and right matrix input ports."""
        return (Port("left"), Port("right"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return the output port declaration."""
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (
            Port(GRAD_LEFT, ValueKind.GRADIENT),
            Port(GRAD_RIGHT, ValueKind.GRADIENT),
            Port(GRAD_LEFT_UNREDUCED, ValueKind.GRADIENT),
            Port(GRAD_RIGHT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        left, right = inputs
        (output,) = outputs
        return (
            reduced_gradient_tensor(left),
            reduced_gradient_tensor(right),
            unreduced_gradient_tensor_matmul(output, left),
            unreduced_gradient_tensor_matmul(output, right),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        left, right = inputs
        (output,) = outputs
        return active_matmul_auxiliary_ports(left, right, output)

    @property
    def backward(self) -> BackwardSpec:
        """Declare both operands as possible backward values."""
        return BackwardSpec(
            supported=True,
            saved_for_backward=("left", "right"),
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Infer the batched matrix-product tensor."""
        left, right = inputs
        if len(left.shape) < 2 or len(right.shape) < 2:
            raise ValueError(
                f"MatMul expects rank >= 2, got {left.shape} and {right.shape}"
            )
        batch = _broadcast_batch(left, right, family=self.family)
        m, left_k = left.shape[-2:]
        right_k, n = right.shape[-2:]
        if left_k != right_k:
            raise ValueError(
                f"incompatible MatMul contracting dimensions: "
                f"{left.shape} @ {right.shape}"
            )
        requires_grad = any(value.requires_grad for value in inputs)
        return (
            Tensor(
                shape=(*batch, m, n),
                semantic_type=left.semantic_type,
                requires_grad=requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        """Save only the operands required by the requested gradients."""
        left, right = inputs
        names: list[str] = []
        if right.requires_grad:
            names.append("left")
        if left.requires_grad:
            names.append("right")
        return tuple(names)

    def forward_flops(self, context: EstimationContext) -> int:
        """Return ``2 * batch * M * K * N`` for concrete dimensions."""
        left = context.tensor_for("left")
        right = context.tensor_for("right")
        output = context.tensor_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        batch = numel(_batch_tensor(output))
        return 2 * batch * left.shape[-2] * left.shape[-1] * right.shape[-1]

    def backward_flops(self, context: EstimationContext) -> int:
        """Return one forward-sized GEMM per requested operand gradient."""
        left = context.tensor_for("left")
        right = context.tensor_for("right")
        output = context.tensor_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flop = 0
        batch = numel(_batch_tensor(output))
        m, k = output.shape[-2], left.shape[-1]
        n = output.shape[-1]
        if left.requires_grad:
            flop += 2 * batch * m * n * k
            flop += _batch_reduction_flops(left, output)
        if right.requires_grad:
            flop += 2 * batch * m * n * k
            flop += _batch_reduction_flops(right, output)
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report forward output allocation and backward gradient aux events."""
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)


__all__ = ["MatMul"]
