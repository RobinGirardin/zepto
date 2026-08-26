"""Invocation context for lowering and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..accounting import AccountingPolicy, PrecisionPolicy
from ..metadata import DType


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Concrete invocation choices for one structural graph lowering pass."""

    phase: str
    hardware: str
    backend: str
    precision: PrecisionPolicy
    accounting: AccountingPolicy
    state: tuple[tuple[str, object], ...] = ()
    implementation_pins: Mapping[str, str] = MappingProxyType({})
    module_implementation_pins: Mapping[str, str] = MappingProxyType({})
    region_implementation_pins: Mapping[str, str] = MappingProxyType({})
    requested_capabilities: frozenset[str] = frozenset()
    allow_fallback: bool = True


def reference_context(
    *,
    phase: str = "forward",
    hardware: str = "generic",
    backend: str = "reference",
    default_dtype: DType = DType.FP32,
    implementation_pins: Mapping[str, str] | None = None,
    module_implementation_pins: Mapping[str, str] | None = None,
    region_implementation_pins: Mapping[str, str] | None = None,
    requested_capabilities: frozenset[str] | None = None,
    allow_fallback: bool = True,
) -> InvocationContext:
    """Build a reference lowering context with FP32 defaults."""
    precision = PrecisionPolicy(default_dtype=default_dtype)
    accounting = AccountingPolicy(precision=precision)
    pins = (
        MappingProxyType(implementation_pins)
        if implementation_pins is not None
        else MappingProxyType({})
    )
    module_pins = (
        MappingProxyType(module_implementation_pins)
        if module_implementation_pins is not None
        else MappingProxyType({})
    )
    region_pins = (
        MappingProxyType(region_implementation_pins)
        if region_implementation_pins is not None
        else MappingProxyType({})
    )
    capabilities = requested_capabilities or frozenset()
    return InvocationContext(
        phase=phase,
        hardware=hardware,
        backend=backend,
        precision=precision,
        accounting=accounting,
        implementation_pins=pins,
        module_implementation_pins=module_pins,
        region_implementation_pins=region_pins,
        requested_capabilities=capabilities,
        allow_fallback=allow_fallback,
    )
