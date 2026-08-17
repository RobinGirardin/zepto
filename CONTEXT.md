# Zepto Cost Estimation

Zepto describes model computations and estimates their resource costs under an explicitly simulated execution context.

## Model structure

**Structural graph**:
The immutable, backend-neutral graph produced by eager PyTorch-like model composition. It records semantic operations, tensors, parameters, and provenance, but is not itself the authoritative execution-cost representation.
_Avoid_: Lowered graph, execution graph

**Module**:
A user-level composition and provenance boundary whose called operations become structural graph nodes. Modules are reportable, but are not computational nodes by default.
_Avoid_: Layer node, component node

**Operation**:
A semantic computational node that maps input tensors to output tensors according to a mathematical operation and a complete semantic and estimation contract.
_Avoid_: Kernel, implementation

**Tensor**:
A graph value with semantic identity, shape/type metadata, and explicit relationships to storage, aliases, parameters, state, and operation ports.
_Avoid_: Buffer, memory allocation

**Provenance**:
Structured origin metadata linking an operation or tensor to its module path, component, source operation, and graph-local identity for attribution and diagnostics.
_Avoid_: Display name, string ID

## Lowering and execution

**Invocation**:
One concrete use of a structural graph, such as a prefill, decode step, or training forward/backward pass, with resolved shapes, state, and policies.
_Avoid_: Horizon

**Lowered graph**:
An immutable, context-specific execution representation derived from a structural graph for one invocation. It is the authoritative input to cost estimation.
_Avoid_: Structural graph, architecture graph

**Lowering**:
The context-driven transformation that selects concrete implementations and maps structural operation and tensor ports to lowered nodes and storage behavior.
_Avoid_: Execution, tracing

**Implementation**:
A concrete execution strategy for a structural operation family, identified independently of the structural operation and selected through lowering.
_Avoid_: Operation, backend

**Fused kernel**:
A lowered execution strategy that combines multiple logical operations without materializing all intermediate outputs in global memory.
_Avoid_: Fused operation

**Resource event**:
An ordered lowered-execution event describing allocation, release, aliasing, saving, persistence, workspace, gradient, or other storage behavior used by peak-memory simulation.
_Avoid_: Memory formula

**State port**:
An explicit invocation boundary for persistent or evolving state, with lifecycle rules describing initialization, reading, updating, retention, and release.
_Avoid_: Persistent tensor role

**Horizon**:
A timeline that composes invocation-specific lowered graphs and simulates repeated use and state transitions, such as KV-cache growth or gradient accumulation.
_Avoid_: Long traversal, giant graph

## Estimation

**Estimation context**:
An immutable typed description of every context-dependent choice that can affect lowering or cost, including hardware, backend, dtype, layout, shapes, phase, state, policies, and implementation requests.
_Avoid_: Global configuration

**Theoretical FLOP**:
A mathematical operation count using Zepto's convention that one multiply-add counts as two FLOPs.
_Avoid_: Runtime instruction count, wall-clock time

**Peak allocated VRAM**:
The maximum live tensor and workspace allocation modeled for an invocation or horizon. Allocator-reserved memory and fragmentation are outside the initial accounting boundary.
_Avoid_: Reserved VRAM, total device memory

**Lowering record**:
The immutable result of lowering, containing the selected implementations, rejected candidates, fallback decisions, context identity, and structural-to-lowered mappings needed for reproducible estimation and reporting.
_Avoid_: Selection log
