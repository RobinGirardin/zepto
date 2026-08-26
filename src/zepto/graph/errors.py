"""Domain errors raised by structural graph and composition APIs."""

class GraphError(ValueError):
    """Base error for invalid structural graph declarations."""


class GraphAlreadyFinalizedError(GraphError):
    """Raised when a finalized builder is mutated."""

    pass


class CrossGraphReferenceError(GraphError):
    """Raised when a graph-owned identity belongs to another graph."""

    pass


class MetadataMismatchError(GraphError):
    """Raised when metadata does not satisfy a port declaration."""

    pass


class UnknownTensorError(GraphError):
    """Raised when a tensor identity is not present."""

    pass


class UnknownParameterError(GraphError):
    """Raised when a parameter identity is not present."""

    pass


class UnknownOperationError(GraphError):
    """Raised when an operation identity is not present."""

    pass


class DuplicatePortError(GraphError):
    """Raised when an operation declares duplicate port names."""

    pass


class PortArityError(GraphError):
    """Raised when port and value counts do not match."""

    pass


class InvalidPortReferenceError(GraphError):
    """Raised when a port reference cannot be resolved."""

    pass


class InvalidOperationOrderError(GraphError):
    """Raised when an operation consumes a later-produced tensor."""

    pass


class InvalidGraphOutputError(GraphError):
    """Raised when a graph output is not a valid graph tensor."""

    pass


class ComposeError(GraphError):
    """Raised when eager module composition is used incorrectly."""

    pass
