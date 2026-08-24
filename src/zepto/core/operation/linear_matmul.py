"""Linear layer matrix multiply: activation × weight parameter."""

from ..metadata import ValueMetadata, TensorRole
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_matmul_auxiliary_ports,
    allocate,
    emit_binary_backward_resource_events,
    numel,
    reduced_gradient_metadata,
    unreduced_gradient_metadata_matmul,
)
from .matmul import _batch_metadata, _batch_reduction_flops, _broadcast_batch
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationError,
    OperationResult,
    ResourceEvent,
)


class LinearMatMul(Operation):
    """Multiply an activation by a weight parameter along the last two dims."""

    @property
    def family(self) -> str:
        return "linear_matmul"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(
                "weight",
                value_kind=ValueKind.PARAMETER,
                metadata=ValueMetadata((), semantic_type="weight"),
            ),
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(GRAD_LEFT, ValueKind.GRADIENT),
            PortSpec(GRAD_RIGHT, ValueKind.GRADIENT),
            PortSpec(GRAD_LEFT_UNREDUCED, ValueKind.GRADIENT),
            PortSpec(GRAD_RIGHT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (activation,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return (
            reduced_gradient_metadata(activation),
            reduced_gradient_metadata(weight),
            unreduced_gradient_metadata_matmul(output, activation),
            unreduced_gradient_metadata_matmul(output, weight),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        (activation,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return active_matmul_auxiliary_ports(activation, weight, output)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input", "weight"),
            gradient_inputs=("output",),
            gradient_outputs=("input", "weight"),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (activation,) = inputs
        (weight,) = parameters
        if len(activation.shape) < 2:
            raise OperationError("linear_matmul requires rank >= 2 activation")
        if len(weight.shape) != 2:
            raise OperationError("linear_matmul weight must be rank 2")
        if activation.shape[-1] != weight.shape[0]:
            raise OperationError(
                f"in_features mismatch: activation {activation.shape[-1]} "
                f"vs weight {weight.shape[0]}"
            )
        batch = _broadcast_batch(activation, weight, family=self.family)
        out_shape = (*batch, activation.shape[-2], weight.shape[1])
        requires_grad = activation.requires_grad or weight.requires_grad
        return (
            ValueMetadata(
                out_shape,
                requires_grad=requires_grad,
                role=TensorRole.ACTIVATION,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        (activation,) = inputs
        (weight,) = parameters
        names: list[str] = []
        if weight.requires_grad:
            names.append("input")
        if activation.requires_grad:
            names.append("weight")
        return tuple(names)

    def forward_flops(self, context: EstimationContext) -> int:
        activation = context.metadata_for("input")
        weight = context.metadata_for("weight")
        output = context.metadata_for("output")
        if activation is None or weight is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'weight', and 'output' ports"
            )
        batch = numel(_batch_metadata(output))
        return 2 * batch * activation.shape[-2] * activation.shape[-1] * weight.shape[-1]

    def backward_flops(self, context: EstimationContext) -> int:
        activation = context.metadata_for("input")
        weight = context.metadata_for("weight")
        output = context.metadata_for("output")
        if activation is None or weight is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'weight', and 'output' ports"
            )
        flop = 0
        batch = numel(_batch_metadata(output))
        m, k = output.shape[-2], weight.shape[0]
        n = output.shape[-1]
        if activation.requires_grad:
            flop += 2 * batch * m * n * k
            flop += _batch_reduction_flops(weight, output)
        if weight.requires_grad:
            flop += 2 * batch * m * n * k
            flop += _batch_reduction_flops(activation, output)
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)
