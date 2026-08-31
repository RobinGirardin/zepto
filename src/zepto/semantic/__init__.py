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
        Concat,
        Cos,
        DeclarationValidator,
        Divide,
        EmbeddingLookup,
        EstimationContext,
        EstimationOperation,
        Exp,
        Gather,
        GreaterThan,
        Identity,
        IncompleteOperationError,
        InferenceBundle,
        InvocationValidator,
        LinearMatMul,
        Log,
        MatMul,
        Materialization,
        MaterializedCausalMask,
        Maximum,
        Minimum,
        Multiply,
        Operation,
        OperationError,
        OperationResult,
        ParameterBias,
        ParameterScale,
        Pow,
        ReduceSum,
        RepeatKV,
        Reshape,
        ResourceEvent,
        ResourceEventKind,
        SemanticOperation,
        Sin,
        Split,
        SquareRoot,
        Subtract,
        Transpose,
        Where,
    )
    from .ports import Port, PortLink, ValueKind

__all__ = [
    "Add",
    "AliasSpec",
    "BackwardSpec",
    "Concat",
    "Cos",
    "DType",
    "DeclarationValidator",
    "Dim",
    "Divide",
    "EmbeddingLookup",
    "EstimationContext",
    "EstimationOperation",
    "Exp",
    "Gather",
    "GreaterThan",
    "Identity",
    "IncompleteOperationError",
    "InferenceBundle",
    "InvocationValidator",
    "LinearMatMul",
    "Log",
    "MatMul",
    "Materialization",
    "MaterializedCausalMask",
    "Maximum",
    "Minimum",
    "Multiply",
    "Operation",
    "OperationError",
    "OperationResult",
    "ParameterBias",
    "ParameterScale",
    "Port",
    "PortContract",
    "PortLink",
    "Pow",
    "ReduceSum",
    "RepeatKV",
    "Reshape",
    "ResourceEvent",
    "ResourceEventKind",
    "SemanticOperation",
    "Shape",
    "Sin",
    "Split",
    "SquareRoot",
    "Subtract",
    "TensorRole",
    "Transpose",
    "ValueKind",
    "Where",
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
