"""Embedding lookup from a weight parameter and token-id indices."""

from ..metadata import ValueMetadata, TensorRole
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, persist_only_events, reduced_gradient_metadata
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_WEIGHT = "grad_weight"


class EmbeddingLookup(Operation):
    """Gather rows from ``weight`` ``(d, V)`` using ``token_ids`` ``(S,)`` → ``(S, d)``."""

    @property
    def family(self) -> str:
        return "embedding_lookup"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("token_ids", metadata=ValueMetadata((), semantic_type="token_ids")),)

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
        return (PortSpec(GRAD_WEIGHT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (_token_ids,) = inputs
        (weight,) = parameters
        return (reduced_gradient_metadata(weight),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
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
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (token_ids,) = inputs
        (weight,) = parameters
        if len(weight.shape) != 2:
            raise ValueError(
                f"embedding_lookup weight must be rank-2 (d, V), got {weight.shape}"
            )
        if len(token_ids.shape) != 1:
            raise ValueError(
                f"embedding_lookup token_ids must be rank-1 (S,), got {token_ids.shape}"
            )
        hidden_size, _vocab = weight.shape
        seq_len = token_ids.shape[0]
        return (
            ValueMetadata(
                (seq_len, hidden_size),
                semantic_type="tensor",
                requires_grad=token_ids.requires_grad or weight.requires_grad,
                role=TensorRole.ACTIVATION,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
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
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["EmbeddingLookup"]
