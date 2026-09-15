"""Invocation context for lowering and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from ..accounting import AccountingPolicy, PrecisionPolicy
from zepto.semantic.metadata import DType

if TYPE_CHECKING:
    from ..runtime.policy import RuntimeOverheadPolicy


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Concrete invocation choices for one graph lowering pass."""

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
    attention_backend: str = "eager"
    allow_fallback: bool = True
    optim_prec: int | None = None
    compute_capability: tuple[int, int] | None = None
    runtime_policy: RuntimeOverheadPolicy | None = None


def reference_invocation(
    *,
    phase: str = "forward",
    hardware: str = "generic",
    backend: str = "reference",
    default_dtype: DType = DType.FP32,
    precision: PrecisionPolicy | None = None,
    optim_prec: int | None = None,
    compute_capability: tuple[int, int] | None = None,
    runtime_policy: RuntimeOverheadPolicy | None = None,
    implementation_pins: Mapping[str, str] | None = None,
    module_implementation_pins: Mapping[str, str] | None = None,
    region_implementation_pins: Mapping[str, str] | None = None,
    requested_capabilities: frozenset[str] | None = None,
    attention_backend: str = "eager",
    state: tuple[tuple[str, object], ...] | None = None,
    allow_fallback: bool = True,
) -> InvocationContext:
    """Build a reference lowering context with FP32 defaults."""
    resolved_precision = precision or PrecisionPolicy(default_dtype=default_dtype)
    accounting = AccountingPolicy(precision=resolved_precision)
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
    invocation_state = state if state is not None else ()
    return InvocationContext(
        phase=phase,
        hardware=hardware,
        backend=backend,
        precision=resolved_precision,
        accounting=accounting,
        state=invocation_state,
        implementation_pins=pins,
        module_implementation_pins=module_pins,
        region_implementation_pins=region_pins,
        requested_capabilities=capabilities,
        attention_backend=attention_backend,
        allow_fallback=allow_fallback,
        optim_prec=optim_prec,
        compute_capability=compute_capability,
        runtime_policy=runtime_policy,
    )
