"""Runtime / software-overhead VRAM policies (cuBLAS workspace, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..horizon.spec import StepKind
from ..lowering.context import InvocationContext

# Constants (identical to Atto)
_CUBLAS_WORKSPACE_SM90_PLUS = 4096 * 8 * 1024
_CUBLAS_WORKSPACE_PRE_SM90 = 4096 * 1024 * 2 + 16 * 1024 * 8


def cublas_workspace_bytes_per_handle(compute_capability: tuple[int, int]) -> int:
    major, _ = compute_capability
    return _CUBLAS_WORKSPACE_SM90_PLUS if major >= 9 else _CUBLAS_WORKSPACE_PRE_SM90


class RuntimeOverheadPolicy(ABC):
    @abstractmethod
    def workspace_bytes(
        self,
        context: InvocationContext,
        *,
        step_kind: StepKind | None = None,
    ) -> int:
        """Return runtime workspace bytes for one invocation."""


@dataclass(frozen=True, slots=True)
class CudaCublasWorkspacePolicy(RuntimeOverheadPolicy):
    compute_capability: tuple[int, int] | None = None

    def workspace_bytes(
        self,
        context: InvocationContext,
        *,
        step_kind: StepKind | None = None,
    ) -> int:
        if context.hardware != "cuda":
            return 0
        cap = (
            self.compute_capability
            if self.compute_capability is not None
            else context.compute_capability
        )
        if cap is None:
            return 0
        per_handle = cublas_workspace_bytes_per_handle(cap)
        handles = _cublas_handle_count(context.phase, step_kind=step_kind)
        return handles * per_handle


class NullRuntimeOverheadPolicy(RuntimeOverheadPolicy):
    def workspace_bytes(self, context, *, step_kind=None) -> int:
        del context, step_kind
        return 0


def _cublas_handle_count(phase: str, *, step_kind: StepKind | None) -> int:
    """Map Zepto phase / horizon step to cuBLAS handle count."""
    if phase in ("backward", "full"):
        return 2
    if step_kind == StepKind.BACKWARD:
        return 2
    return 1


def runtime_workspace_bytes(
    context: InvocationContext,
    *,
    step_kind: StepKind | None = None,
    policy: RuntimeOverheadPolicy | None = None,
) -> int:
    """Resolve runtime workspace for a context (default: CUDA cuBLAS policy)."""
    resolved = policy or context.runtime_policy or CudaCublasWorkspacePolicy()
    return resolved.workspace_bytes(context, step_kind=step_kind)
