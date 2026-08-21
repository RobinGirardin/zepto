"""Lowering pass: structural graph to lowered invocation graph."""

from .context import InvocationContext, reference_context
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
    "reference_context",
    "select_implementation",
]
