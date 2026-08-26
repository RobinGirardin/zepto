"""Parameter snapshots stored on the structural graph."""

from zepto.compose.values import Parameter, Tensor

__all__ = ["Parameter"]


def parameter_as_tensor(parameter: Parameter) -> Tensor:
    """Project a stored parameter into a structural tensor for inference."""
    return parameter.as_tensor()
