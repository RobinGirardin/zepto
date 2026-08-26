"""Lowering pass: graph to lowered invocation graph."""

from .context import InvocationContext, reference_invocation
from .defaults import DEFAULT_REGISTRY
from .errors import LoweringError
from .transform import lower
from .registry import (
    ImplementationDescriptor,
    ImplementationSelection,
    LoweringRegistry,
    select_implementation,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "ImplementationDescriptor",
    "ImplementationSelection",
    "InvocationContext",
    "LoweringError",
    "LoweringRegistry",
    "lower",
    "reference_invocation",
    "select_implementation",
]
