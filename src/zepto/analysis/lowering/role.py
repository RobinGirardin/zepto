"""Accounting-role inference shared by lowering and estimation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from zepto.compose.values import Tensor
from zepto.graph.ids import EdgeId
from zepto.semantic.metadata import TensorRole
from zepto.semantic.ports import ValueKind
from ..resolved import ResolvedValue
from .context import InvocationContext

if TYPE_CHECKING:
    from zepto.graph.graph import Graph


PortDirection = Literal["input", "output", "parameter", "auxiliary"]


@dataclass(frozen=True, slots=True)
class RoleContext:
    """Facts needed to infer an accounting role for one tensor."""

    graph: Graph | None = None
    edge_id: EdgeId | None = None
    port_name: str | None = None
    port_direction: PortDirection | None = None
    port_value_kind: ValueKind | None = None
    explicit_role: TensorRole | None = None


def infer_accounting_role(ctx: RoleContext) -> TensorRole:
    """Infer the storage/accounting role from graph and port context."""
    if ctx.explicit_role is not None:
        return ctx.explicit_role
    if ctx.graph is not None and ctx.edge_id is not None and ctx.edge_id in ctx.graph.inputs:
        return TensorRole.INPUT
    if ctx.port_direction == "parameter":
        return TensorRole.PARAMETER
    if ctx.port_direction == "auxiliary":
        if ctx.port_value_kind is ValueKind.GRADIENT:
            if ctx.port_name is not None and ctx.port_name.endswith("_unreduced"):
                return TensorRole.WORKSPACE
            return TensorRole.GRADIENT
        return TensorRole.AUXILIARY
    return TensorRole.ACTIVATION


def resolve_for_accounting(
    tensor: Tensor,
    *,
    role_ctx: RoleContext,
    context: InvocationContext,
) -> ResolvedValue:
    """Infer role and resolve dtype for estimation."""
    role = infer_accounting_role(role_ctx)
    dtype = context.accounting.resolve_dtype(tensor, role=role)
    return ResolvedValue(tensor=tensor, role=role, dtype=dtype)


def resolve_lowered_edge(
    tensor: Tensor,
    *,
    role_ctx: RoleContext,
    context: InvocationContext,
) -> tuple[Tensor, TensorRole]:
    """Infer role and return a tensor copy with resolved dtype for lowering."""
    resolved = resolve_for_accounting(tensor, role_ctx=role_ctx, context=context)
    return replace(resolved.tensor, dtype=resolved.dtype), resolved.role
