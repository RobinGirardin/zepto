"""Bilinear 2-D grid resize operation declaration."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from zepto.semantic.metadata import PortContract
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


@dataclass(frozen=True, slots=True)
class BilinearResize2D(Operation):
    """Resize a 2-D grid ``(H, W, C)`` to ``(out_h, out_w, C)``."""

    out_h: int
    out_w: int

    @property
    def family(self) -> str:
        return "bilinear_resize_2d"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return ()

    @property
    def parameter_ports(self) -> tuple[Port, ...]:
        return (
            Port(
                "input",
                value_kind=ValueKind.PARAMETER,
                contract=PortContract(semantic_type="position_embedding"),
            ),
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=False)

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        if len(parameters) != 1:
            raise ValueError("bilinear_resize_2d requires one input parameter")
        (input_tensor,) = parameters
        if len(input_tensor.shape) != 3:
            raise ValueError(
                f"bilinear_resize_2d expects rank-3 (H, W, C), got {input_tensor.shape}"
            )
        if self.out_h <= 0 or self.out_w <= 0:
            raise ValueError("out_h and out_w must be positive")
        return (
            Tensor(
                shape=(self.out_h, self.out_w, input_tensor.shape[2]),
                dtype=input_tensor.dtype,
                semantic_type=input_tensor.semantic_type,
                requires_grad=input_tensor.requires_grad,
                persistent=input_tensor.persistent,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        input_tensor = context.tensor_for("input")
        if input_tensor is None:
            for name, value in context.port_values:
                if name == "input" and value.tensor is not None:
                    input_tensor = value.tensor
                    break
        if input_tensor is None or len(input_tensor.shape) != 3:
            return 0
        channels = input_tensor.shape[2]
        return 4 * self.out_h * self.out_w * channels

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["BilinearResize2D"]
