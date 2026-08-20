"""Batched matrix multiplication operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class MatMul(Operation):
    """Multiply tensors along their final two dimensions.

    Leading dimensions represent batch dimensions and must currently match.
    """

    @property
    def family(self) -> str:
        """Return the stable matrix multiplication family name."""
        return "matmul"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the left and right matrix input ports.

        Their metadata remains ``None`` until graph construction binds the
        concrete input tensor metadata.
        """
        return (PortSpec("left"), PortSpec("right"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the output port declaration.

        Its metadata remains ``None`` until graph construction binds the
        inferred product metadata.
        """
        return (PortSpec("output"),)

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
        self, inputs: tuple[TensorMetadata, ...]
    ) -> tuple[TensorMetadata, ...]:
        """Infer the batched matrix-product metadata."""
        left, right = inputs
        if len(left.shape) < 2 or len(right.shape) < 2:
            raise ValueError(
                "MatMul expects tensors with rank of at least two, not"
                f"{left.shape} and {right.shape}"
            )
        if len(left.shape) != len(right.shape):
            raise ValueError(
                f"MatMul batch ranks must match, got shapes {left.shape} and {right.shape}"
            )
        left_batch = left.shape[:-2]
        right_batch = right.shape[:-2]
        if left_batch != right_batch:
            raise ValueError(
                f"incompatible MatMul batch dimensions: {left.shape} @ {right.shape}"
            )
        m, left_k = left.shape[-2:]
        right_k, n = right.shape[-2:]
        if left_k != right_k:
            raise ValueError(
                f"incompatible MatMul contracting dimensions: {left.shape} @ {right.shape}"
            )
        require_grad = any(value.require_grad for value in inputs)
        return (
            TensorMetadata(
                shape=(*left_batch, m, n),
                semantic_type=left.semantic_type,
                require_grad=require_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        """Save only the operands required by the requested gradients.

        The gradient of ``left`` contracts the output gradient with
        ``right``, and the gradient of ``right`` contracts it with ``left``.
        """
        left, right = inputs
        names: list[str] = []
        if right.require_grad:
            names.append("left")
        if left.require_grad:
            names.append("right")
        return tuple(names)

    def forward_flops(self, context: EstimationContext) -> int:
        """Return ``2 * batch * M * K * N`` for concrete dimensions."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        if left is None or right is None:
            raise ValueError(
                "Estimation context must provide both a 'left' and 'right' port"
            )
        batch = numel(TensorMetadata(left.shape[:-2]))
        return 2 * batch * left.shape[-2] * left.shape[-1] * right.shape[-1]

    def backward_flops(self, context: EstimationContext) -> int:
        """Return one forward-sized GEMM per requested operand gradient."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        output = context.metadata_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flop = 0
        batch = numel(TensorMetadata(left.shape[:-2]))
        if left.require_grad:
            b_transpose_flop = 0 # View
            flop += (
                2
                * batch
                * output.shape[-2]
                * output.shape[-1]
                * right.shape[-2]  # -2, instead of -1 due to B^T
                + b_transpose_flop
            )
        if right.require_grad:
            a_transpose_flop = 0 # View
            flop += (
                2
                * batch
                * output.shape[-2]
                * output.shape[-1]
                * left.shape[-1]  # contracting dimension K of A^T @ dY
                + a_transpose_flop
            )
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the product output."""
        return allocate(result)

__all__ = ["MatMul"]
