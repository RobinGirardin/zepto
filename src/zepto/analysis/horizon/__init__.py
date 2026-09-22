"""Horizon simulation and accounting."""

from .account import account_horizon_flops, account_horizon_memory
from .records import HorizonSimulation, InvocationRecord
from .simulate import (
    InputsFn,
    ModuleFn,
    inputs_from_shape,
    inputs_from_token_ids,
    inputs_from_token_ids_and_labels,
    simulate_horizon,
)
from .spec import (
    ConvStateConfig,
    HorizonSpec,
    HorizonStep,
    KVConfig,
    RecurrentStateConfig,
    StepKind,
    decode_step,
    optimizer_step,
)
from .state import (
    Conv1DState,
    GradAccumState,
    KVCacheState,
    OptimizerState,
    RecurrentScanState,
    StatePortRegistry,
    StateSnapshot,
)

__all__ = [
    "Conv1DState",
    "ConvStateConfig",
    "GradAccumState",
    "HorizonSimulation",
    "HorizonSpec",
    "HorizonStep",
    "InputsFn",
    "InvocationRecord",
    "KVCacheState",
    "KVConfig",
    "RecurrentScanState",
    "RecurrentStateConfig",
    "ModuleFn",
    "OptimizerState",
    "StatePortRegistry",
    "StateSnapshot",
    "StepKind",
    "account_horizon_flops",
    "account_horizon_memory",
    "decode_step",
    "inputs_from_shape",
    "inputs_from_token_ids",
    "inputs_from_token_ids_and_labels",
    "optimizer_step",
    "simulate_horizon",
]
