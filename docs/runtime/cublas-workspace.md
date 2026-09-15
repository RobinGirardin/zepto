# CUDA cuBLAS workspace (runtime overhead)

Zepto models PyTorch's per-handle cuBLAS workspace as **runtime overhead**, separate from graph-local transient storage in `MemoryBreakdown.workspace`.

## Formulas

From PyTorch `CublasHandlePool.cpp`, function `parseChosenWorkspaceSize()` ([commit `71b66af50dd18422e9ed558d83e3582d14aa1639`](https://github.com/pytorch/pytorch/blob/71b66af50dd18422e9ed558d83e3582d14aa1639/aten/src/ATen/cuda/CublasHandlePool.cpp)):

**SM90+ (major ≥ 9), per handle:**

```
4096 * 8 * 1024  →  33_554_432 bytes
```

**Pre-SM90, per handle:**

```
4096 * 1024 * 2 + 16 * 1024 * 8  →  8_519_680 bytes
```

## Handle counts

| Scenario | Handles |
|----------|---------|
| Inference / prefill / decode / micro-forward | 1 |
| Backward or merged train (`phase="full"`) | 2 |
| Optimizer step | 0 |
| Non-CUDA hardware | 0 |
| Missing `compute_capability` | 0 |

Train-like phases charge **two** handles because PyTorch may hold separate cuBLAS workspaces for forward and backward (often on different threads).

## Zepto API

Configure on `InvocationContext` only (not on `estimate()` kwargs):

```python
from dataclasses import replace

from zepto.analysis import (
    CudaCublasWorkspacePolicy,
    NullRuntimeOverheadPolicy,
    estimate,
    reference_invocation,
)

ctx = reference_invocation(
    hardware="cuda",
    compute_capability=(8, 0),
    runtime_policy=CudaCublasWorkspacePolicy(compute_capability=(8, 0)),
)
report = estimate(graph, ctx)
```

When `runtime_policy` is omitted, `runtime_workspace_bytes()` defaults to `CudaCublasWorkspacePolicy()`, which reads `context.compute_capability`. Without capability, runtime workspace is **0**.

## Breakdown fields

| Field | Contents |
|-------|----------|
| `breakdown.workspace` | Graph-local temps (unreduced matmul/VJP, `TensorRole.WORKSPACE` edges) |
| `breakdown.runtime_workspace` | cuBLAS handle pool for the invocation |
| `peak_live_bytes` | Simulator peak + `runtime_workspace` |

`OptimizerPolicy.workspace_bytes` is unrelated — it covers Adam scratch, not cuBLAS.
