"""2-D convolution for ViT-style patch embedding."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from zepto.semantic.metadata import PortContract
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _spatial_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


@dataclass(frozen=True, slots=True)
class Conv2d(Operation):
    """Channel-last conv2d: input ``(H, W, C_in)`` → ``(H', W', C_out)``."""

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0

    @property
    def family(self) -> str:
        return "conv2d"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

    @property
    def parameter_ports(self) -> tuple[Port, ...]:
        return (
            Port(
                "weight",
                value_kind=ValueKind.PARAMETER,
                contract=PortContract(semantic_type="weight"),
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
        (inp,) = inputs
        if len(inp.shape) != 3:
            raise ValueError("conv2d expects rank-3 input (H, W, C_in)")
        h, w, c = inp.shape
        if c != self.in_channels:
            raise ValueError(
                f"conv2d in_channels mismatch: expected {self.in_channels}, got {c}"
            )
        out_h = _spatial_out(h, self.kernel_size, self.stride, self.padding)
        out_w = _spatial_out(w, self.kernel_size, self.stride, self.padding)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("conv2d output spatial size must be positive")
        return (
            Tensor(
                shape=(out_h, out_w, self.out_channels),
                dtype=inp.dtype,
                requires_grad=inp.requires_grad,
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
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide 'output' port")
        spatial = numel(output) // self.out_channels
        k = self.kernel_size
        return 2 * spatial * k * k * self.in_channels * self.out_channels

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["Conv2d"]
