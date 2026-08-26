"""Structured graph-origin metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provenance:
    """Structured origin information for a graph operation or tensor."""

    module_path: tuple[str, ...]
    component_type: str | None
    operation_family: str
    operation_instance: int
    source_label: str | None = None

    def __post_init__(self) -> None:
        """Validate operation family and instance identity fields."""
        if not self.operation_family:
            raise ValueError("Operation family cannot be empty")
        if self.operation_instance < 0:
            raise ValueError("Operation instance cannot be negative")
