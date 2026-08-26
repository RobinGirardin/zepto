"""User-facing eager graph composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .values import Parameter, Tensor

if TYPE_CHECKING:
    from .context import Compose, Module, compose_graph

__all__ = [
    "Compose",
    "Module",
    "Parameter",
    "Tensor",
    "compose_graph",
]

_CONTEXT_NAMES = frozenset({"Compose", "Module", "compose_graph"})


def __getattr__(name: str):
    if name in _CONTEXT_NAMES:
        from . import context

        return getattr(context, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
