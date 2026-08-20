"""Core graph, operation, metadata, and composition APIs."""

from .composition import (
    GraphCompositionContext,
    GraphTensor,
    Module,
    build_graph,
)
from .errors import *
from .functional import add, identity, matmul, multiply, relu, reshape, split, transpose
from .graph import StructuralGraph, StructuralGraphBuilder
from .ids import *
from .metadata import (
    Dim,
    Shape,
    TensorMetadata,
    metadata_compatible,
)
from .operation import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    EstimationOperation,
    IncompleteOperationError,
    Materialization,
    OperationError,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
    SemanticOperation,
    StructuralOperation,
    Operation,
    Add,
    Identity,
    MatMul,
    Multiply,
    ReLU,
    Reshape,
    Split,
    Transpose,
)
from .parameter import Parameter
from .ports import PortRef, PortSpec, ValueKind
from .provenance import Provenance
from .tensor import Tensor

__all__ = [
    "Dim",
    "AliasSpec",
    "BackwardSpec",
    "EstimationContext",
    "EstimationOperation",
    "IncompleteOperationError",
    "GraphCompositionContext",
    "GraphTensor",
    "MetadataMismatchError",
    "Materialization",
    "Module",
    "OperationError",
    "Operation",
    "OperationResult",
    "PortRef",
    "PortSpec",
    "Provenance",
    "ResourceEvent",
    "ResourceEventKind",
    "SemanticOperation",
    "Shape",
    "StructuralGraph",
    "StructuralGraphBuilder",
    "StructuralOperation",
    "Tensor",
    "TensorMetadata",
    "ValueKind",
    "build_graph",
    "identity",
    "matmul",
    "add",
    "multiply",
    "Add",
    "Identity",
    "MatMul",
    "Multiply",
    "ReLU",
    "Reshape",
    "Split",
    "Transpose",
    "reshape",
    "relu",
    "split",
    "transpose",
    "metadata_compatible",
]
