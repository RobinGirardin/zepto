"""Lowering, estimation, and resource accounting."""

from .accounting import AccountingPolicy, PrecisionPolicy
from .estimation import estimate
from .lowered import LoweredEdge, LoweredGraph, LoweredNode
from .resolved import ResolvedValue
from .lowering.role import RoleContext, infer_accounting_role
from .lowering import (
    DEFAULT_REGISTRY,
    ImplementationDescriptor,
    ImplementationSelection,
    InvocationContext,
    LoweringError,
    LoweringRegistry,
    RegionImplementationDescriptor,
    RegionImplementationSelection,
    Region,
    discover_regions,
    lower,
    reference_invocation,
    select_implementation,
    select_region_implementation,
)

__all__ = [
    "AccountingPolicy",
    "DEFAULT_REGISTRY",
    "ImplementationDescriptor",
    "ImplementationSelection",
    "InvocationContext",
    "LoweredEdge",
    "LoweredGraph",
    "LoweredNode",
    "LoweringError",
    "LoweringRegistry",
    "PrecisionPolicy",
    "RegionImplementationDescriptor",
    "RegionImplementationSelection",
    "ResolvedValue",
    "RoleContext",
    "Region",
    "discover_regions",
    "estimate",
    "infer_accounting_role",
    "lower",
    "reference_invocation",
    "select_implementation",
    "select_region_implementation",
]
