"""Lowering pass: graph to lowered invocation graph."""

from .context import InvocationContext, reference_invocation
from .defaults import DEFAULT_REGISTRY
from .discovery import discover_regions
from .errors import LoweringError
from .plan import LoweringPlan, LoweringStep, build_lowering_plan
from .region import (
    PatternConstraint,
    PatternMatchRule,
    ProvenanceMatchRule,
    StructuralRegion,
)
from .transform import lower
from .registry import (
    ImplementationDescriptor,
    ImplementationSelection,
    LoweringRegistry,
    RegionImplementationDescriptor,
    RegionImplementationSelection,
    select_implementation,
    select_region_implementation,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "ImplementationDescriptor",
    "ImplementationSelection",
    "InvocationContext",
    "LoweringError",
    "LoweringPlan",
    "LoweringStep",
    "LoweringRegistry",
    "PatternConstraint",
    "PatternMatchRule",
    "ProvenanceMatchRule",
    "RegionImplementationDescriptor",
    "RegionImplementationSelection",
    "StructuralRegion",
    "build_lowering_plan",
    "discover_regions",
    "lower",
    "reference_invocation",
    "select_implementation",
    "select_region_implementation",
]
