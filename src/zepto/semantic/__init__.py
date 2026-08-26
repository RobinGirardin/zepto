"""Semantic operation declarations and port contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metadata import (
    DType,
    Dim,
    Shape,
    PortContract,
    TensorRole,
    contract_satisfied,
)

if TYPE_CHECKING:
    from .operations import (
        Add,
        AliasSpec,
        BackwardSpec,
        DeclarationValidator,
        Divide,
        EstimationContext,
        EstimationOperation,
        Identity,
        IncompleteOperationError,
        InferenceBundle,
        InvocationValidator,
        LinearMatMul,
        MatMul,
        Materialization,
        Maximum,
        Minimum,
        Multiply,
        Operation,
        OperationError,
        OperationResult,
        Reshape,
        ResourceEvent,
        ResourceEventKind,
        SemanticOperation,
        Split,
        SquareRoot,
        Subtract,
        Transpose,
    )
    from .ports import Port, PortLink, ValueKind

__all__ = [
    "Add",
    "AliasSpec",
    "BackwardSpec",
    "DType",
    "DeclarationValidator",
    "Dim",
    "Divide",
    "EstimationContext",
    "EstimationOperation",
    "Identity",
    "IncompleteOperationError",
    "InferenceBundle",
    "InvocationValidator",
    "LinearMatMul",
    "MatMul",
    "Materialization",
    "Maximum",
    "Minimum",
    "Multiply",
    "Operation",
    "OperationError",
    "OperationResult",
    "Port",
    "PortContract",
    "PortLink",
    "Reshape",
    "ResourceEvent",
    "ResourceEventKind",
    "SemanticOperation",
    "Shape",
    "Split",
    "SquareRoot",
    "Subtract",
    "TensorRole",
    "Transpose",
    "ValueKind",
    "contract_satisfied",
]

_METADATA = frozenset(
    {"DType", "Dim", "Shape", "PortContract", "TensorRole", "contract_satisfied"}
)
_PORT_NAMES = frozenset({"Port", "PortLink", "ValueKind"})
_OPERATION_NAMES = frozenset(__all__) - _METADATA - _PORT_NAMES


def __getattr__(name: str):
    if name in _PORT_NAMES:
        from . import ports

        return getattr(ports, name)
    if name in _OPERATION_NAMES:
        from . import operations

        return getattr(operations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
