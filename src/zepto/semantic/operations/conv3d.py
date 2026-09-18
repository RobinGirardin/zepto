"""3-D convolution for video tubulet patch embedding."""

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
class Conv3d(Operation):
    """Channel-last conv3d: input ``(T, H, W, C_in)`` → ``(T', H', W', C_out)``."""

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: tuple[int, int, int] = (0, 0, 0)

    @property
    def family(self) -> str:
        return "conv3d"

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
        if len(inp.shape) != 4:
            raise ValueError("conv3d expects rank-4 input (T, H, W, C_in)")
        t, h, w, c = inp.shape
        if c != self.in_channels:
            raise ValueError(
                f"conv3d in_channels mismatch: expected {self.in_channels}, got {c}"
            )
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding
        out_t = _spatial_out(t, kt, st, pt)
        out_h = _spatial_out(h, kh, sh, ph)
        out_w = _spatial_out(w, kw, sw, pw)
        if min(out_t, out_h, out_w) <= 0:
            raise ValueError("conv3d output spatial size must be positive")
        return (
            Tensor(
                shape=(out_t, out_h, out_w, self.out_channels),
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
        kt, kh, kw = self.kernel_size
        return 2 * spatial * kt * kh * kw * self.in_channels * self.out_channels

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["Conv3d"]
