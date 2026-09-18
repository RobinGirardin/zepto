"""Optimizer policy contracts for horizon training steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    """Describe optimizer state and update-time workspace."""

    name: str
    state_bytes_per_parameter: int
    workspace_bytes: int = 0
    flops_per_parameter: int = 0


AdamW = OptimizerPolicy(
    name="adamw", state_bytes_per_parameter=8, flops_per_parameter=7
)
SGD = OptimizerPolicy(name="sgd", state_bytes_per_parameter=0, flops_per_parameter=2)


def optimizer_update_flops(
    policy: OptimizerPolicy,
    parameter_bytes: int,
    *,
    bytes_per_element: int,
) -> int:
    """Elementwise optimizer update FLOPs from stored parameter bytes.

    Mixed dtypes across parameters are not supported yet; callers should derive
    ``bytes_per_element`` from the first parameter in the lowered graph.
    """
    if policy.flops_per_parameter == 0 or parameter_bytes == 0:
        return 0
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
    num_elements = parameter_bytes // bytes_per_element
    return num_elements * policy.flops_per_parameter
