"""Embedding lookup from a weight parameter and token-id indices."""

from zepto.compose.values import Tensor
from zepto.semantic.metadata import PortContract
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, activation_grad_events, reduced_gradient_tensor
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_WEIGHT = "grad_weight"


class EmbeddingLookup(Operation):
    """Gather rows from ``weight`` ``(d, V)`` using token indices.

    Supports ``token_ids`` ``(S,)`` → ``(S, d)`` and batched ``(B, S)`` → ``(B, S, d)``.
    """

    @property
    def family(self) -> str:
        return "embedding_lookup"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("token_ids"),)

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

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (Port(GRAD_WEIGHT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (_token_ids,) = inputs
        (weight,) = parameters
        return (reduced_gradient_tensor(weight),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        (_token_ids,) = inputs
        (weight,) = parameters
        if weight.requires_grad:
            return (GRAD_WEIGHT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("token_ids",),
            gradient_inputs=("output",),
            gradient_outputs=("weight",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (token_ids,) = inputs
        (weight,) = parameters
        if len(weight.shape) != 2:
            raise ValueError(
                f"embedding_lookup weight must be rank-2 (d, V), got {weight.shape}"
            )
        hidden_size, _vocab = weight.shape
        if len(token_ids.shape) == 1:
            out_shape = (token_ids.shape[0], hidden_size)
        elif len(token_ids.shape) == 2:
            batch, seq_len = token_ids.shape
            out_shape = (batch, seq_len, hidden_size)
        else:
            raise ValueError(
                "embedding_lookup token_ids must be rank-1 (S,) or rank-2 (B, S), "
                f"got {token_ids.shape}"
            )
        return (
            Tensor(
                shape=out_shape,
                dtype=weight.dtype,
                semantic_type="tensor",
                requires_grad=token_ids.requires_grad or weight.requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        (_token_ids,) = inputs
        (weight,) = parameters
        if weight.requires_grad:
            return ("token_ids",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["EmbeddingLookup"]
