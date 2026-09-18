---
status: accepted
---
# Model memory with resource events and compose horizons

Lowered implementations will expose ordered resource events over separate tensor and storage identities, including allocation, alias/view, save-for-backward, release, persistence, workspace, and gradient behavior. Peak allocated tensor/workspace VRAM is computed from these events; allocator-reserved memory and fragmentation are outside the initial target.

One lowered graph represents one concrete invocation. A horizon composes invocation graphs over a timeline and simulates state-port lifecycles such as KV-cache growth, optimizer-state retention, and gradient accumulation. This keeps execution-specific memory semantics explicit without turning an arbitrarily long workload into one graph.

Training horizons include an optimizer step as a timeline boundary: it advances optimizer state ports via a parameter-only stub lowered graph without composing or lowering the training module again. Optimizer rows reuse the prior step’s structural graph reference for bookkeeping only.
