from functools import partial

from .composition import GraphTensor, GraphCompositionContext, OperationSpec
from .metadata import TensorMetadata
from .ports import PortSpec


def identity(value: GraphTensor) -> GraphTensor:
    """Record an identity operation for a symbolic graph value.

    Args:
        value: Symbolic input value.

    Returns:
        A symbolic output with the same metadata as ``value``.
    """
    context = _require_context()
    spec = OperationSpec(
        family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        infer_outputs=lambda inputs: inputs,
    )
    return context.apply(spec, value)  # type: ignore[return-value]


def infer_linear_outputs(
    inputs: tuple[TensorMetadata, ...],
    *,
    output_features: int,
) -> tuple[TensorMetadata, ...]:
    """Infer the output metadata for a linear projection.

    Args:
        inputs: Exactly one input metadata value.
        output_features: Size of the projected final dimension.

    Returns:
        One output metadata value with the final dimension replaced.

    Raises:
        ValueError: If the input arity, rank, or feature size is invalid.
    """
    if len(inputs) != 1:
        raise ValueError(f"Linear expects one input, got {len(inputs)}")
    if output_features < 0:
        raise ValueError("Linear output_features cannot be negative")

    input_metadata = inputs[0]
    if not input_metadata.shape:
        raise ValueError("Linear input must have at least one dimension")

    return (
        TensorMetadata(
            shape=(*input_metadata.shape[:-1], output_features),
            semantic_type=input_metadata.semantic_type,
        ),
    )


def linear(
    value: GraphTensor,
    *,
    output_features: int,
    weight=None,
    bias=None,
) -> GraphTensor:
    """Record a backend-neutral linear operation.

    Args:
        value: Symbolic input whose final dimension is projected.
        output_features: Size of the output feature dimension.
        weight: Optional graph parameter identity for the weight.
        bias: Optional graph parameter identity for the bias.

    Returns:
        A symbolic output with the projected shape.

    Raises:
        ValueError: If ``value`` has no dimensions.
    """
    context = _require_context()
    spec = OperationSpec(
        family="linear",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        infer_outputs=partial(
            infer_linear_outputs,
            output_features=output_features,
        ),
    )
    parameters = tuple(
        parameter for parameter in (weight, bias) if parameter is not None
    )
    return context.apply(spec, value, parameters=parameters)  # type: ignore[return-value]


def _require_context() -> GraphCompositionContext:
    """Return the active composition context or raise a usage error."""
    context = GraphCompositionContext.current()
    if context is None:
        raise RuntimeError("Functional operations require an active graph context")
    return context
