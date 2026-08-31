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
A reusable semantic operation declaration that maps input tensors and
parameters to output tensors according to a mathematical operation and
complete semantic and estimation behavior. Calling an operation during
composition records a node in the structural graph.
_Avoid_: Kernel, implementation

**Tensor**:
A user-facing description of a data-flow value during model composition —
shape, dtype, and gradient or persistence flags. Once registered, it carries
an edge identity linking it to the structural graph.
_Avoid_: GraphTensor, flow handle, buffer

**Parameter**:
A user-facing description of a persistent model weight declared during module
construction. Once registered, it carries a parameter identity and is stored
on the graph, not as a data-flow edge.
_Avoid_: GraphParameter, weight handle, parameter tensor node, ParameterAsset

**Edge**:
An immutable graph connection carrying a frozen tensor description,
producer and consumer port links, and optional storage identity.
_Avoid_: Graph tensor, flow edge, buffer

**Node**:
One recorded occurrence of an operation in the structural graph, binding
ports to edges and parameters with provenance and invocation metadata.
_Avoid_: Structural operation, layer node, kernel

**Port**:
A named input, output, parameter, or auxiliary slot declared on an operation.
_Avoid_: PortSpec

**Port contract**:
A partial specification of what a port accepts — shape, dtype, or semantic
type. Unset fields and an empty shape are wildcards. Accounting role is not
part of the structural value; it is inferred during analysis.
_Avoid_: Port metadata, ValueMetadata, bound port metadata

**Resolved value**:
An analysis-time pairing of a structural tensor with its inferred accounting
role and resolved dtype. It is ephemeral and is not stored on the structural
graph.
_Avoid_: ValueMetadata, bound port metadata

**PortLink**:
A reference to a specific port on a node, used to connect edges through
the graph.
_Avoid_: PortRef, port reference

**Compose**:
The eager tracing context in which module and operation calls record structure
into an immutable graph. Supports a simple tier (automatic root input and
output registration) and an explicit tier for advanced composition.
_Avoid_: Graph composition context, tracing session

**Auxiliary tensor**:
A tensor used internally by an operation or its backward computation, such as a
normalization statistic or activation mask. It is a graph value with storage
and lifetime semantics, but is not necessarily a module's public output.
_Avoid_: Opaque saved state

**Public output**:
A tensor returned through an operation or module's user-facing result.
_Avoid_: Every operation tensor

**Saved backward value**:
An ordinary graph tensor that one operation invocation retains because its
backward computation depends on it. The operation's backward declaration
lists the ports it is allowed to save; each invocation selects only the
subset required by the requested gradients.
_Avoid_: Context blob, implicit saved lifetime

**Provenance**:
Structured origin metadata linking an operation or tensor to its module path, component, source operation, and graph-local identity for attribution and diagnostics.
_Avoid_: Display name, string ID

## Lowering and execution

**Invocation**:
One concrete use of a structural graph, such as a prefill, decode step, or training forward/backward pass, with concrete shapes, state, and policies. Different shape scenarios use independently rebuilt structural graphs.
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

**Region**:
A lowering-time candidate for fusing multiple graph nodes into one implementation leaf. Carries boundary edges, internal node ids, parameters, and module provenance. Not stored on the graph at composition time.
_Avoid_: Structural region, fused module

**Resource event**:
An ordered lowered-execution event describing allocation, release, aliasing,
persistence, workspace, or other storage behavior used by memory accounting.
_Avoid_: Memory formula

**State port**:
An explicit invocation boundary for persistent or evolving state, with lifecycle rules describing initialization, reading, updating, retention, and release.
_Avoid_: Persistent tensor role

**Horizon**:
A timeline that composes invocation-specific lowered graphs and simulates repeated use and state transitions, such as KV-cache growth or gradient accumulation.
_Avoid_: Long traversal, giant graph

## Estimation

**Estimation context**:
An immutable typed description of every context-dependent choice that can affect lowering or cost, including hardware, backend, dtype, layout, concrete shapes, phase, state, policies, and implementation requests.
_Avoid_: Global configuration

**Backend profile**:
A scenario label on an estimation context that names which whole-invocation costing profile is being modeled (for example reference, vLLM serve, or HuggingFace eager). It narrows compatible implementation variants; it is not executable backend code and is not a per-kernel override.
_Avoid_: CUDA backend, runtime backend

**Attention backend**:
The estimation-context choice for how grouped-query attention is costed — for example eager decomposition versus a FlashAttention-style fused leaf. Defaults to eager when unspecified.
_Avoid_: PyTorch SDPA dispatcher, kernel source

**Region kind**:
The fusion category for a region and its registered implementations (for example RMSNorm or grouped-query attention). Variants within one kind share discovery rules; backend-specific cost leaves differ by implementation identity, not by kind.
_Avoid_: Backend folder, module type

**Theoretical FLOP**:
A mathematical operation count using Zepto's convention that one multiply-add counts as two FLOPs.
_Avoid_: Runtime instruction count, wall-clock time

**Peak allocated VRAM**:
The maximum live tensor and workspace allocation modeled for an invocation or horizon. Allocator-reserved memory and fragmentation are outside the initial accounting boundary.
_Avoid_: Reserved VRAM, total device memory

**Lowering record**:
The immutable result of lowering, containing the selected implementations, rejected candidates, fallback decisions, context identity, and structural-to-lowered mappings needed for reproducible estimation and reporting.
_Avoid_: Selection log

**Memory accounting**:
Analysis of a lowered graph's storage behavior, including total allocation,
peak live allocation, persistent storage, workspace, and related breakdowns.
_Avoid_: VRAM formula

**Memory report**:
The result of memory accounting for an invocation or horizon, containing both
sum-all and peak measurements and their attribution.
_Avoid_: Single VRAM number

**Optimizer policy**:
A contract describing optimizer state and update-time resource behavior
independently of model operations.
_Avoid_: Optimizer operation

**Operation validator**:
A focused check for one boundary of an operation contract: its declaration,
concrete invocation result, or graph binding.
_Avoid_: Cost accounting
