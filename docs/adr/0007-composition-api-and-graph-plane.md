---
status: accepted
---
# Composition API and graph plane refactor

Zepto will unify the user-facing composition model and clarify the immutable graph plane. The goal is PyTorch-familiar ergonomics (`y = model(Tensor(...))`) with a single recording gate, flat user types, and explicit graph artifacts (`Edge`, `Node`) distinct from composition values.

## User-facing composition

**Callable operations.** `Operation.__call__` is the sole user-facing recording gate. It resolves the active `Compose` context and records a graph node. `forward` methods call operations directly and never reference context.

**Context-free `forward`.** Module `forward` expresses structure only — operation calls and nested modules. Provenance and graph recording happen in `Module.__call__` and `Operation.__call__` via the active compose context.

**Flat `Tensor` and `Parameter`.** Users declare values with top-level fields (`shape`, `dtype`, `requires_grad`, `trainable`, …). `GraphTensor` and `GraphParameter` are retired. Each type carries an optional private identity (`_edge_id`, `_parameter_id`) set at registration time. Instances are frozen; registration returns a new instance with the identity filled in.

**Two-tier compose API.**

- *Simple (default):* root `Module.__call__` lazily registers any input whose `_edge_id` is unset; `build()` auto-marks the root return value as graph output when no outputs were explicitly marked. No `ctx.input` or `ctx.mark_output` required for the common single-input, single-output path.
- *Explicit:* `ctx.input(...)` and `ctx.mark_output(...)` remain for multi-output graphs, partial composition, and tests.

Lazy input registration applies **only at the root module boundary**. An unregistered `Tensor` passed inside `forward` is an error, not an implicit graph input.

**Zero-ceremony entry point.** `compose_graph(module, inputs=...)` (successor to `build_graph`) wraps the simple tier: open context, construct module, call with inputs, finalize graph.

## Graph plane

**`Edge` replaces graph `Tensor`.** An edge is immutable once recorded. It holds a frozen `Tensor` snapshot (semantic description), connectivity (`producer`, `consumers` as `PortLink`s), optional `storage_id`, and provenance. Semantic description lives on the embedded `Tensor`; connectivity lives on the `Edge`.

**`Node` replaces `StructuralOperation`.** A node is one recorded operation occurrence: port bindings to edge and parameter identities, provenance, and invocation result metadata. It references the reusable operation declaration but is a distinct graph artifact.

**`Parameter` in the graph.** Registered parameters are stored in `graph.parameters` as frozen `Parameter` snapshots keyed by `ParameterId`. There is no separate graph-parameter wrapper type.

**Internal metadata contracts.** Superseded by ADR-0008: compose and graph use `Tensor` / `Parameter` directly; ports declare `PortContract`; analysis infers accounting role.

## Naming

- `GraphCompositionContext` → `Compose`; `GraphCompositionError` → `ComposeError`
- `build_graph` → `compose_graph`
- `StructuralGraph` / `StructuralGraphBuilder` → `Graph` / `GraphBuilder` (`graph/graph.py`)
- `PortSpec` → `Port`; `PortRef` → `PortLink`; `PortLink.operation_id` → `node_id`
- `OperationId` / `TensorId` removed; public API uses `NodeId` / `EdgeId` only
- `LoweredOperation` / `LoweredTensor` → `LoweredNode` / `LoweredEdge` (`LoweredEdge.tensor: Tensor`)
- `reference_context` → `reference_invocation`
- Functional wrappers removed; recording is via callable operations only

Bound port metadata duplication on nodes is removed in ADR-0008.

## Module layout

Reorganize into four packages (names may adjust during migration):

- `compose` — context, `Module`, `Tensor`, `Parameter`, `compose_graph`
- `graph` — `Edge`, `Node`, `Graph`, `GraphBuilder`
- `semantic` — operation declarations, ports, inference contracts
- `analysis` — lowering, estimation, accounting (unchanged consumers of immutable graph)

## Migration phases

1. Add `Operation.__call__`; rewire functional wrappers. *(done)*
2. Hide direct `context().apply` from user-facing API and docs; deprecate functional wrappers. *(done)*
3. Rename ports; introduce unified `Tensor` / `Parameter` with identity fields. *(done)*
4. Introduce `Edge` / `Node`; retire graph-level `Tensor` and `StructuralOperation` naming. *(done)*
5. Reorganize module layout. *(done)*

Phases are independently shippable; behavior of immutable graph validation, lowering, and estimation must remain correct after each phase.

Import from `zepto.compose`, `zepto.graph`, `zepto.semantic`, and `zepto.analysis`. The top-level `zepto` namespace re-exports the public API.

## Non-goals

- Pydantic for validation (deferred; gain too small).
- Symbolic shapes (unchanged; see ADR-0005).
- Operations forced into `Module.__init__` (stateless ops may be constructed inline or bound once — user choice).

## Rationale

The previous model split recording across `Module.__call__`, `context().apply`, and functional wrappers, and overloaded `Tensor` between composition handles and graph storage. Unifying call semantics, flattening user types, and naming graph artifacts explicitly reduces conceptual load without changing the immutable structural / lowered graph split established in ADR-0001.
