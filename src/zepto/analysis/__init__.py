"""Lowering, estimation, and resource accounting."""

from .accounting import AccountingPolicy, PrecisionPolicy
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
from .reports import CostReport, FlopReport, MemoryReport
from .memory import account_memory
from .flops import account_flops
from .estimation import estimate

__all__ = [
    "AccountingPolicy",
    "CostReport",
    "DEFAULT_REGISTRY",
    "FlopReport",
    "ImplementationDescriptor",
    "ImplementationSelection",
    "InvocationContext",
    "LoweredEdge",
    "LoweredGraph",
    "LoweredNode",
    "LoweringError",
    "LoweringRegistry",
    "MemoryReport",
    "PrecisionPolicy",
    "RegionImplementationDescriptor",
    "RegionImplementationSelection",
    "ResolvedValue",
    "RoleContext",
    "Region",
    "account_flops",
    "account_memory",
    "discover_regions",
    "estimate",
    "infer_accounting_role",
    "lower",
    "reference_invocation",
    "select_implementation",
    "select_region_implementation",
]
