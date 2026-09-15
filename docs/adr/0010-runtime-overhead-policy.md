# ADR-0010: Runtime overhead policies

## Status

Accepted

## Context

Graph resource events model tensor lifetimes inside a lowered graph. PyTorch also holds persistent library workspace (cuBLAS handle pools) for the entire CUDA invocation. That overhead is outside the graph event model but affects peak VRAM.

## Decision

- Add a `RuntimeOverheadPolicy` contract in `zepto.analysis.runtime`, mirroring the optimizer policy pattern (ADR-0006).
- Expose `compute_capability` and optional `runtime_policy` on `InvocationContext` only.
- Add `MemoryBreakdown.runtime_workspace` distinct from graph-local `workspace`.
- Apply runtime bytes **after** `ResourceEventSimulator`: add to `peak_live_bytes` and `sum_all_bytes` per invocation.
- Default policy: `CudaCublasWorkspacePolicy()` → 0 without explicit capability; non-CUDA hardware → 0.

## Consequences

- Peak and sum-all increase when callers pass CUDA capability; default offline estimates unchanged.
- Horizon steps pass `StepKind` so optimizer steps charge 0 handles while backward steps charge 2.
- ROCm/MPS workspace policies deferred; use `NullRuntimeOverheadPolicy`.
