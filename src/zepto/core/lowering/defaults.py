"""Default lowering registry populated with built-in implementations."""

from .implementations import register_defaults
from .registry import LoweringRegistry

DEFAULT_REGISTRY = LoweringRegistry()
register_defaults(DEFAULT_REGISTRY)
