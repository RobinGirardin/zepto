"""Shared context lookup for functional operation wrappers."""

from ..composition import GraphCompositionContext


def context() -> GraphCompositionContext:
    """Return the active graph composition context.

    Raises:
        RuntimeError: If no graph composition context is active.
    """
    active = GraphCompositionContext.current()
    if active is None:
        raise RuntimeError("Functional operations require an active graph context")
    return active
