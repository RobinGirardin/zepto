"""User-facing composition values for eager graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from zepto.graph.errors import ComposeError
from zepto.graph.ids import EdgeId, ParameterId
from zepto.semantic.metadata import DType, Shape


def _validate_shape(shape: Shape) -> None:
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError("Tensor dimensions must be non-negative integers")


@dataclass(frozen=True, slots=True)
class Tensor:
    """Description of a data-flow value during model composition."""

    shape: Shape
    dtype: DType | None = None
    requires_grad: bool = False
    persistent: bool = False
    semantic_type: str = "tensor"
    _edge_id: EdgeId | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")
        if self.dtype is not None and not isinstance(self.dtype, DType):
            raise ValueError("dtype must be a DType when supplied")
        if not isinstance(self.persistent, bool):
            raise ValueError("persistent must be an explicit boolean")
        if not isinstance(self.requires_grad, bool):
            raise ValueError("requires_grad must be an explicit boolean")

    @property
    def id(self) -> EdgeId:
        """Return the graph edge identity for a registered tensor."""
        if self._edge_id is None:
            raise ComposeError("Tensor is not registered in the graph")
        return self._edge_id

    def registered(self, edge_id: EdgeId) -> Tensor:
        """Return a registered copy bound to a graph edge identity."""
        return replace(self, _edge_id=edge_id)

    def as_unregistered(self) -> Tensor:
        """Return a structural copy without a graph edge identity."""
        return replace(self, _edge_id=None)


@dataclass(frozen=True, slots=True)
class Parameter:
    """Description of a persistent model weight during composition."""

    shape: Shape
    dtype: DType | None = None
    trainable: bool = True
    semantic_type: str = "parameter"
    _parameter_id: ParameterId | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_shape(self.shape)
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")
        if self.dtype is not None and not isinstance(self.dtype, DType):
            raise ValueError("dtype must be a DType when supplied")
        if not isinstance(self.trainable, bool):
            raise ValueError("trainable must be an explicit boolean")

    @property
    def id(self) -> ParameterId:
        """Return the graph parameter identity for a registered parameter."""
        if self._parameter_id is None:
            raise ComposeError("Parameter is not registered in the graph")
        return self._parameter_id

    def registered(self, parameter_id: ParameterId) -> Parameter:
        """Return a registered copy bound to a graph parameter identity."""
        return replace(self, _parameter_id=parameter_id)

    def as_tensor(self) -> Tensor:
        """Project this parameter into a structural tensor for inference."""
        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            requires_grad=self.trainable,
            persistent=True,
            semantic_type=self.semantic_type,
        )
