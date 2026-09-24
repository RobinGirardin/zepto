"""Training boundary costs applied outside graph lowering."""

from __future__ import annotations

from dataclasses import dataclass

from ..lowering.context import InvocationContext
from ..optimizer import OptimizerPolicy


@dataclass(frozen=True, slots=True)
class TrainingBoundaryCost:
    optimizer_state_bytes: int
    optimizer_update_flops: int
    optimizer_workspace_bytes: int


def training_boundary_cost(
    *,
    policy: OptimizerPolicy,
    trainable_elements: int,
    context: InvocationContext,
) -> TrainingBoundaryCost:
    return TrainingBoundaryCost(
        optimizer_state_bytes=policy.state_bytes(
            trainable_elements=trainable_elements, context=context
        ),
        optimizer_update_flops=policy.update_flops(
            trainable_elements=trainable_elements, context=context
        ),
        optimizer_workspace_bytes=policy.update_workspace_bytes(
            trainable_elements=trainable_elements, context=context
        ),
    )
