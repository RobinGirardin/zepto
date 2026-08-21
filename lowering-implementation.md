# Lowering Implementation Plan

Compiled from the VRAM accounting design discussions. This document describes
what lowering is, what code to add, how implementation selection works, and
how specialized implementations (notably ReLU-as-Maximum) differ from identity
lowering.

Derived from `operation-vram-plan.md`, `operation-vram-next-steps.md`,
`CONTEXT.md`, and ADRs 0001, 0002, 0004, 0006.

---

## 1. Current framework state

After the completed next-steps work (Steps 1–7), Zepto has a clean
**structural** pipeline and a deliberately incomplete **execution** pipeline:

```text
Module.forward → GraphCompositionContext → StructuralGraphBuilder → StructuralGraph
                                              ↑ validate (no memory/FLOP)
```

### What exists today

| Layer | What it does |
|-------|----------------|
| `StructuralGraph` | Immutable semantic graph: tensors, operations, storage IDs, saved-for-backward PortRefs |
| `Operation` | Backend-neutral declaration: ports, inference, aliases, backward spec, **estimation hooks** |
| `EstimationContext` | Per-invocation metadata for estimators (`port_metadata`, optional `precision`) |
| `AccountingPolicy` | Resolves dtype → bytes (evaluation side, not yet wired into a pipeline) |
| Validators | Declaration / invocation / graph checks — **explicitly do not** call `resource_events()` or FLOP hooks |

### What is missing

- No `InvocationContext` (concrete invocation choices: phase, backend, policies)
- No `LoweredGraph` / `LoweredOperation` / `LoweredTensor`
- No `lower()` transform
- No place where `resource_events()`, `forward_flops()`, and `backward_flops()` are actually invoked and validated
- No `estimate()` composition API (§9–10 in the VRAM plan; depends on lowering)

Operations already *declare* lowered behavior (`Identity` emits `ALIAS`, `Maximum`
emits `ALLOCATE`, etc.), but nothing consumes those declarations yet. Lowering
fills that gap.

---

## 2. Conceptual model: two graphs, one transform

Per ADR-0001 and `CONTEXT.md`, lowering is the boundary between **meaning** and
**execution strategy**:

```text
StructuralGraph  +  InvocationContext
        │
        ▼
     lower()
        │
        ▼
   LoweredGraph  ──→  account_memory() / account_flops()  (next phase)
```

Per ADR-0006, lowering and accounting are separate:

```text
structural graph → lower → lowered graph
                         ├→ account_memory → MemoryReport
                         └→ account_flops   → FlopReport
```

`estimate()` may compose these as a convenience API but does not replace the
individual entry points.

### Rules lowering must enforce

1. **Structural graph is never mutated.** Lowering produces a new immutable record.
2. **One lowered graph = one invocation** (forward pass, backward pass, prefill step, etc.).
3. **Storage identity is the accounting key.** View aliases from the structural graph (`Materialization.VIEW`) must become `ALIAS` events referencing existing storage, not fresh allocations.
4. **Dtype is resolved here.** Structural metadata may leave `dtype=None`; lowering applies `PrecisionPolicy` from context and writes concrete `TensorMetadata` copies onto lowered tensors.
5. **Implementation selection is explicit.** Even identity lowering (1 structural op → 1 lowered op) must record *which* implementation was chosen and why (ADR-0002: registry, rejected candidates, pins).
6. **Resource events become authoritative.** They move from "methods on `Operation` that structural validation ignores" to "frozen tuples on `LoweredOperation` that memory accounting consumes."

### Domain terminology

| Term | Meaning |
|------|---------|
| **Structural graph** | Backend-neutral semantic source of truth |
| **Lowered graph** | Immutable, context-specific execution representation for one invocation |
| **Lowering** | Context-driven transform selecting implementations and mapping ports to storage behavior |
| **Implementation** | Concrete execution strategy for an operation *family*, independent of the structural `Operation` class |
| **Lowering record** | Immutable result with mappings, selections, rejected candidates, and fallback decisions |

---

## 3. New modules and types

### 3.1 `src/zepto/core/lowered.py` — lowered records

```python
@dataclass(frozen=True, slots=True)
class LoweredTensor:
    id: str                          # lowered-local identity
    structural_tensor_id: TensorId | None
    metadata: TensorMetadata         # dtype/role resolved
    storage_id: str                  # string form of StorageId for accounting
    workspace: bool = False


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    id: str
    structural_operation_id: OperationId
    implementation: str              # selected implementation id
    input_tensors: tuple[str, ...]   # lowered tensor ids
    output_tensors: tuple[str, ...]
    auxiliary_tensors: tuple[str, ...]   # empty until §3 auxiliary ports land
    resource_events: tuple[ResourceEvent, ...]
    forward_flops: int
    backward_flops: int


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    tensors: Mapping[str, LoweredTensor]
    operations: tuple[LoweredOperation, ...]
    context: InvocationContext
    tensor_map: Mapping[TensorId, str]       # structural → lowered
    operation_map: Mapping[OperationId, str]
    selections: tuple[ImplementationSelection, ...]
```

`LoweredGraph` is the **only** input memory and FLOP accounting should accept.

### 3.2 `src/zepto/core/lowering/` — context, registry, transform

#### `InvocationContext`

Superset of what `EstimationContext` carries today:

```python
@dataclass(frozen=True, slots=True)
class InvocationContext:
    phase: str                       # "forward" | "backward"
    hardware: str                    # e.g. "generic"
    backend: str                     # e.g. "reference"
    precision: PrecisionPolicy
    accounting: AccountingPolicy     # wraps precision; used for byte resolution
    state: tuple[tuple[str, object], ...] = ()
    implementation_pins: Mapping[str, str] = ()   # family → impl id
    requested_capabilities: frozenset[str] = frozenset()
    allow_fallback: bool = True
```

Relationship to `EstimationContext`:

- `InvocationContext` = user-facing, graph-wide choices
- `EstimationContext` = per-operation slice built **from** invocation context during lowering
- Do not merge into one type; lowering translates between them

#### `ImplementationDescriptor`

```python
@dataclass(frozen=True, slots=True)
class ImplementationDescriptor:
    id: str                    # "maximum/naive", "maximum/relu-mask"
    family: str                # must match Operation.family
    priority: int = 0          # higher wins when auto-selecting
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()   # hard requirements from context
```

`id` is namespaced: `"<family>/<variant>"`.

#### `ImplementationSelection` (ADR-0002 reproducibility)

```python
@dataclass(frozen=True, slots=True)
class ImplementationSelection:
    structural_operation_id: OperationId
    chosen: ImplementationDescriptor
    rejected: tuple[tuple[ImplementationDescriptor, str], ...]
    reason: str                # "pinned", "highest_priority", "only_candidate"
```

#### `Implementation` protocol

An implementation is **not** an `Operation`. It receives an already-validated
structural node and produces a `LoweredOperation`.

```python
class Implementation(Protocol):
    @property
    def descriptor(self) -> ImplementationDescriptor: ...

    def compatible(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> str | None:
        """Return None if compatible, else rejection reason."""

    def lower(
        self,
        structural: StructuralOperation,
        graph: StructuralGraph,
        context: InvocationContext,
        tensor_map: Mapping[TensorId, str],
        *,
        estimation: EstimationContext,
    ) -> LoweredOperation:
        ...
```

- **`compatible()`** — hard requirements (backend tag, capability flags, pattern match).
- **`lower()`** — emits events/FLOPs and wires lowered tensor ids; does **not** re-run `infer_outputs()`.
- **`estimation`** — built once by `lower()` from resolved port metadata.

#### `LoweringRegistry`

```python
class LoweringRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, Implementation] = {}
        self._by_family: dict[str, list[Implementation]] = {}

    def register(self, implementation: Implementation) -> None:
        descriptor = implementation.descriptor
        if descriptor.id in self._by_id:
            raise ValueError(f"Duplicate implementation id {descriptor.id!r}")
        self._by_id[descriptor.id] = implementation
        self._by_family.setdefault(descriptor.family, []).append(implementation)

    def lookup(
        self,
        family: str,
        context: InvocationContext,
    ) -> tuple[Implementation, ImplementationSelection]:
        ...

    def get(self, impl_id: str) -> Implementation | None:
        return self._by_id.get(impl_id)

    def candidates(self, family: str) -> tuple[Implementation, ...]:
        return tuple(self._by_family.get(family, ()))
```

Default registry at module load:

```python
DEFAULT_REGISTRY = LoweringRegistry()
# populated by register_defaults(DEFAULT_REGISTRY)
```

Users pass `registry=` into `lower()` only when they need custom impls.

#### Main entry point

```python
def lower(
    graph: StructuralGraph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> LoweredGraph:
    ...
```

#### Internal helpers (private in `lowering/`)

- `_resolve_tensor_metadata(tensor, parameter, context) -> TensorMetadata`
- `_structural_storage_id(tensor) -> str`
- `_select_implementation(family, context, registry) -> tuple[Implementation, ImplementationSelection]`
- `_build_estimation_context(operation, inputs, outputs, context) -> EstimationContext`
- `_validate_lowered_events(events, operation) -> None`
- `_emit_saved_tensor_events(...)` — or centralized SAVE synthesis in `lower()`
- `_ensure_lowered_tensor(out_tid, graph, context, tensor_map) -> str`

### 3.3 `src/zepto/core/estimation.py` — composition stub

Full `CostReport` waits for memory/FLOP modules (§9–10):

```python
@dataclass(frozen=True, slots=True)
class CostReport:
    memory: MemoryReport
    flops: FlopReport
    lowered: LoweredGraph


def estimate(
    graph: StructuralGraph,
    context: InvocationContext,
    *,
    memory_policy: MemoryAccountingPolicy | None = None,
) -> CostReport:
    lowered = lower(graph, context)
    return CostReport(
        memory=account_memory(lowered, memory_policy),
        flops=account_flops(lowered),
        lowered=lowered,
    )
```

For the lowering-only milestone, this file may export only context builders and
re-export `lower`.

---

## 4. Package layout

```text
src/zepto/core/
├── lowering/
│   ├── __init__.py          # lower(), DEFAULT_REGISTRY, InvocationContext
│   ├── registry.py          # LoweringRegistry, ImplementationDescriptor, Selection
│   ├── context.py           # InvocationContext builder helpers
│   ├── errors.py            # LoweringError
│   └── implementations/
│       ├── __init__.py      # register_identity_defaults, register_specialized
│       ├── identity.py      # IdentityImplementation
│       └── maximum.py       # ReLUMaskImplementation (later)
├── lowered.py               # LoweredGraph, LoweredOperation, LoweredTensor
└── operation/               # unchanged — structural declarations only
```

Keep `operation/` free of backend/implementation code (ADR-0002 boundary).

---

## 5. The `lower()` algorithm

For each structural operation in eager order:

1. **Resolve input metadata** — read bound port metadata from `StructuralOperation` / graph tensors; merge with `InvocationContext.precision`.
2. **Select implementation** — registry lookup by `operation_family`; honor pins; record rejected alternatives (see §6).
3. **Build estimation context**:

```python
EstimationContext(
    phase=context.phase,
    port_metadata=(("left", resolved_left), ("right", resolved_right), ...),
    precision=context.precision,
)
```

4. **Call implementation** (not structural hooks directly, except via identity wrapper):

```python
impl, selection = _select_implementation(structural, graph, context, registry)
lowered_op = impl.lower(structural, graph, context, tensor_map, estimation=estimation)
```

5. **Materialize lowered tensors** for outputs (orchestrator, shared by all impls):
   - `VIEW` alias → reuse structural `storage_id`, emit `ALIAS`
   - Fresh storage → new lowered storage id, emit `ALLOCATE`
   - Register each output in `tensor_map`
6. **Emit SAVE events** for saved-for-backward ports (identity impl or centralized in `lower()` based on structural `saved_for_backward` PortRefs).
7. **Append `LoweredOperation`** with frozen events and FLOP counts.

Graph-level inputs/parameters become `LoweredTensor` entries at the start (roles:
`INPUT`, `PARAMETER`).

---

## 6. Implementation selection

Selection happens inside `lower()` per structural node:

```text
StructuralOperation.operation_family
        │
        ▼
LoweringRegistry.lookup(family, context)
        │
        ▼
Implementation.lower(node, graph, context, tensor_map)
        │
        ▼
LoweredOperation(implementation="maximum/relu-mask", resource_events=..., ...)
```

### Selection algorithm

```python
def _select_implementation(
    structural: StructuralOperation,
    graph: StructuralGraph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Implementation, ImplementationSelection]:
    family = structural.operation_family
    candidates = registry.candidates(family)

    if not candidates:
        raise LoweringError(f"No implementations registered for family {family!r}")

    rejected: list[tuple[ImplementationDescriptor, str]] = []

    # 1. Exact pin from context
    if family in context.implementation_pins:
        pin = context.implementation_pins[family]
        impl = registry.get(pin)
        if impl is None:
            raise LoweringError(f"Pinned implementation {pin!r} not registered")
        reason = impl.compatible(structural, graph, context)
        if reason is not None:
            raise LoweringError(f"Pinned {pin!r} incompatible: {reason}")
        return impl, ImplementationSelection(
            structural.id, impl.descriptor, (), "pinned"
        )

    # 2. Filter by compatibility
    viable = []
    for impl in candidates:
        reason = impl.compatible(structural, graph, context)
        if reason is None:
            viable.append(impl)
        else:
            rejected.append((impl.descriptor, reason))

    if not viable:
        if not context.allow_fallback:
            raise LoweringError(
                f"No compatible implementation for {family!r}; "
                f"rejected: {rejected}"
            )
        # fallback policy: document behavior (e.g. fail or warn)

    # 3. Priority order (stable sort by priority desc, then id)
    viable.sort(key=lambda i: (-i.descriptor.priority, i.descriptor.id))
    chosen = viable[0]

    return chosen, ImplementationSelection(
        structural.id,
        chosen.descriptor,
        tuple(rejected),
        "highest_priority" if len(viable) > 1 else "only_candidate",
    )
```

---

## 7. Identity implementations (bootstrap)

ADR-0001: the first pass is identity lowering. One default implementation per
family delegates to the structural `Operation`'s estimation hooks.

```python
@dataclass(frozen=True, slots=True)
class IdentityImplementation:
    """Wrap one structural Operation as a 1:1 lowered implementation."""

    operation: Operation
    descriptor: ImplementationDescriptor

    def compatible(self, structural, graph, context) -> str | None:
        if structural.operation_family != self.descriptor.family:
            return "family mismatch"
        return None

    def lower(self, structural, graph, context, tensor_map, *, estimation):
        op = structural.declaration or self.operation
        result = structural.result
        assert result is not None

        events = op.resource_events(estimation, result)
        forward = op.forward_flops(estimation)
        backward = op.backward_flops(
            replace(estimation, phase="backward")
        )

        output_ids = tuple(
            _ensure_lowered_tensor(out_tid, graph, context, tensor_map)
            for out_tid in structural.output_tensors
        )
        input_ids = tuple(tensor_map[tid] for tid in structural.input_tensors)

        return LoweredOperation(
            id=f"{structural.id}:0",
            structural_operation_id=structural.id,
            implementation=self.descriptor.id,
            input_tensors=input_ids,
            output_tensors=output_ids,
            auxiliary_tensors=(),
            resource_events=_append_save_events(
                events, structural, graph, tensor_map
            ),
            forward_flops=forward,
            backward_flops=backward,
        )
```

Registration for all shipped ops:

```python
def register_identity_defaults(registry: LoweringRegistry) -> None:
    for operation in (
        Add(), Subtract(), Multiply(), Divide(), MatMul(),
        Identity(), Reshape(), Transpose(), Split(),
        Maximum(), Minimum(), SquareRoot(),
    ):
        registry.register(
            IdentityImplementation(
                operation=operation,
                descriptor=ImplementationDescriptor(
                    id=f"{operation.family}/identity",
                    family=operation.family,
                    priority=0,
                ),
            )
        )
```

This gives a working `lower()` immediately without bespoke impl classes per op.

---

## 8. Specialized implementations

Specialized impls **replace execution behavior** while preserving structural
meaning. They do **not** subclass `Operation`.

### Pattern A: Same family, different resource/FLOP behavior

Example: ReLU as a lowered variant of `Maximum` (see §9).

### Pattern B: Fusion — many structural ops → one lowered op

Future work. The registry API must not assume 1:1 only. A fusion entry would
carry a region matcher rather than a simple family string. ADR-0001 requires
explicit port-level mappings.

---

## 9. ReLU mask implementation (`maximum/relu-mask`)

### Semantic equivalence

For `right ≡ 0`:

```text
output = max(left, 0)          # structural public output (port "output")
mask   = left > 0              # lowering-only auxiliary (boolean or 0/1)
dL/dleft = dL/doutput * mask   # backward uses mask only; dL/dright = 0
```

### Behavioral contrast

| Aspect | `Maximum` / identity impl | `ReLUMaskImplementation` |
|--------|---------------------------|---------------------------|
| Saved for backward | `("left", "right")` — two full operand tensors | One **`mask`** auxiliary only |
| Backward FLOPs | Up to `2 × numel` (both operand grads) | `numel` (one masked multiply on `left`) |
| Forward extra outputs | None (only `"output"`) | **`mask`** — lowered-only, not a structural port |
| Right operand memory | Charged if saved | Never saved (constant zero) |
| Structural `Operation` | Unchanged | Unchanged |

The structural node may still have `result.saved_for_backward == ("left", "right")`
from `Maximum.saved_for_backward()`. **`ReLUMaskImplementation` does not use
that for accounting** — it replaces the event sequence and FLOP counts on the
`LoweredOperation`.

### Where the mask lives

Auxiliary ports are not on the structural graph yet. The mask is **lowering-only**:

- Registered on `LoweredOperation.auxiliary_tensors` as `("{operation_id}:mask",)`
- `LoweredTensor` with `structural_tensor_id=None`
- `role=TensorRole.AUXILIARY`, `semantic_type="relu_mask"`
- Produced and consumed within the same lowered node's forward/backward lifecycle

### Pattern gate in `compatible()`

```python
def compatible(self, structural, graph, context) -> str | None:
    if structural.operation_family != "maximum":
        return "not maximum"

    left_id, right_id = structural.input_tensors
    right = graph.tensor(right_id)

    if not _is_zero_operand(right, graph):
        return "right operand is not provably zero"

    return None
```

Practical `_is_zero_operand` rules:

1. **Constant-zero tensor** — graph input or parameter with `semantic_type="constant_zero"` (or a dedicated `Zero` op later).
2. **Provenance tag** — a future `relu(x)` wrapper desugars to `maximum(x, zero)` and tags provenance with `"relu"`.
3. **Explicit pin** — `context.implementation_pins["maximum"] == "maximum/relu-mask"`.

Structural validation stays generic; the lowering impl encodes the ReLU specialization.

### `ReLUMaskImplementation.lower()` sketch

```python
def lower(self, structural, graph, context, tensor_map, *, estimation):
    left_id, right_id = structural.input_tensors
    output_id = structural.output_tensors[0]

    left_meta = _resolved(graph.tensor(left_id), context)
    output_meta = _resolved(graph.tensor(output_id), context)
    n = numel(output_meta)

    lowered_out = tensor_map[output_id]
    lowered_mask = f"{structural.id}:mask"

    mask_meta = replace(
        output_meta,
        semantic_type="relu_mask",
        role=TensorRole.AUXILIARY,
        dtype=DType.BOOL,
        requires_grad=False,
        persistent=False,
    )

    auxiliary = (lowered_mask,)

    if context.phase == "forward":
        forward_flops = n
        backward_flops = 0
        events = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask),
            ResourceEvent(ResourceEventKind.SAVE, lowered_mask),
        )
    else:
        forward_flops = n
        backward_flops = n if left_meta.requires_grad else 0
        events = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask),
            ResourceEvent(ResourceEventKind.SAVE, lowered_mask),
            ResourceEvent(
                ResourceEventKind.RELEASE, lowered_mask, phase="backward"
            ),
        )

    return LoweredOperation(
        id=f"{structural.id}:0",
        structural_operation_id=structural.id,
        implementation="maximum/relu-mask",
        input_tensors=(tensor_map[left_id],),   # right elided (constant zero)
        output_tensors=(lowered_out,),
        auxiliary_tensors=auxiliary,
        resource_events=events,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
    )
```

Key behaviors:

1. **Do not save `left` or `right`.** Save **`mask` only** via `SAVE`.
2. **Mask lifecycle:** forward `ALLOCATE` + `SAVE`; backward consumes mask, then `RELEASE`.
3. **`input_tensors` drops `right`** on the lowered op (constant, zero storage cost).
4. **FLOPs:** forward = `numel(output)` (compare); backward = `numel` if `left.requires_grad` else 0.

Event lifecycle:

```text
forward:  ALLOCATE(output) → ALLOCATE(mask) → SAVE(mask)
backward: (read mask) → masked multiply → RELEASE(mask)
```

### What stays unchanged on `Maximum(Operation)`

Do **not** modify the structural class for ReLU:

```python
# maximum.py — stays as-is
saved_for_backward(...) -> ("left", "right")  when any grad
BackwardSpec.saved_for_backward = ("left", "right")
resource_events(...) -> allocate(result)
```

Optional later: a `relu(x)` functional that desugars to `maximum(x, zero)` with
provenance `"relu"` to make `compatible()` reliable without pins.

### Registration

```python
RELU_MASK = ReLUMaskImplementation(
    descriptor=ImplementationDescriptor(
        id="maximum/relu-mask",
        family="maximum",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
    ),
)

def register_maximum_implementations(registry: LoweringRegistry) -> None:
    registry.register(RELU_MASK)
    registry.register(IdentityImplementation(Maximum(), ...))
```

Priority 10 beats `maximum/identity` (0) when `compatible()` passes. Generic
`maximum(a, b)` with non-zero `b` fails ReLU `compatible()` and falls through
to identity.

---

## 10. Registration patterns

### A. Central bootstrap (recommended for v1)

```python
# lowering/__init__.py
DEFAULT_REGISTRY = LoweringRegistry()
register_identity_defaults(DEFAULT_REGISTRY)
register_specialized(DEFAULT_REGISTRY)
```

### B. Explicit register at import time (plugins / research)

```python
from zepto.core.lowering import DEFAULT_REGISTRY
DEFAULT_REGISTRY.register(MyCustomMatMul())
```

Prefer an explicit `register_reference_backend()` if backends are added later.

### C. Decorator sugar (optional)

```python
@register_implementation(
    id="matmul/tiled",
    family="matmul",
    priority=5,
    capabilities={"tiled"},
)
class TiledMatMul:
    ...
```

---

## 11. Changes to existing code

### Operations (`src/zepto/core/operation/*.py`)

Mostly **unchanged** for identity lowering. Enhancements during lowering work:

| Change | Why |
|--------|-----|
| Richer event helpers in `helpers.py` | Add `alias(...)`, `save(...)`, `release(...)` for implementations to reuse |
| SAVE synthesis | Today only outputs are allocated; saved PortRefs need explicit `SAVE` at lowering (per-op or centralized) |
| View ops already correct | `Identity` → `ALIAS`; arithmetic → `allocate()` |
| Separate `Implementation` classes | ReLU, fused matmul — **not** new `Operation` subclasses |

### Validators — no rollback

`InvocationValidator` correctly skips FLOP/events. Add **`LoweredOperationValidator`**
(checks event tuple, FLOPs ≥ 0, event targets exist, alias/save targets valid).

### `StructuralGraph` / `composition.py` — unchanged

Graph construction stays semantic-only. Users do not call `lower()` during
`build_graph()`.

### Exports (`core/__init__.py`)

Add: `InvocationContext`, `LoweredGraph`, `LoweredOperation`, `LoweredTensor`,
`lower`, `LoweringRegistry`, `ImplementationDescriptor`, `ImplementationSelection`.
Later: `estimate`, `CostReport`.

---

## 12. API flow modifications

### Composition API (additive)

**Today:**

```python
graph = build_graph(MyModule(), inputs)
```

**After lowering:**

```python
graph = build_graph(MyModule(), inputs)
ctx = InvocationContext(
    phase="forward",
    hardware="generic",
    backend="reference",
    precision=PrecisionPolicy(default_dtype=DType.FP32),
    accounting=AccountingPolicy(precision=...),
)
lowered = lower(graph, ctx)
# later:
report = estimate(graph, ctx)
```

`build_graph` does not change. Lowering is a second pass.

### Operation method call schedule

| Method | When called today | After lowering |
|--------|-------------------|----------------|
| `infer_result()` | Graph construction | Unchanged |
| `validate_result()` | Graph construction | Unchanged (still no events/FLOPs) |
| `resource_events()` | Never during construction | Called inside identity/specialized `lower()` |
| `forward_flops()` / `backward_flops()` | Tests call directly | Called inside `lower()`; stored on `LoweredOperation` |

---

## 13. Relationship: `Operation` vs `Implementation`

| Concern | Structural `Operation` | Lowered `Implementation` |
|---------|------------------------|----------------------------|
| Port semantics, shape inference | `infer_outputs`, `backward` | never overrides |
| Default cost/events | `resource_events`, `forward_flops` | identity impl delegates here |
| Specialized cost/events | same methods (unused if not selected) | specialized impl overrides in `lower()` |
| Registration | exported from `operation/__init__.py` | registered on `LoweringRegistry` |

---

## 14. Dependencies on unfinished work

| Item | Impact on lowering |
|------|-------------------|
| Auxiliary ports (§3) | `LoweredOperation.auxiliary_tensors` used for mask etc.; structural aux ports deferred |
| Optimizer policies (§11) | Optimizer state/events injected during lowering or post-pass |
| ReLU-as-lowered-Maximum | First non-identity implementation |
| Broadcasting normalization | May introduce `Broadcast` as lowering artifact only |

None block **identity lowering** for the existing primitive set.

---

## 15. Tests — `tests/test_lowering.py`

- Identity lowering: 1 structural op → 1 lowered op, same port wiring
- Structural-to-lowered tensor/operation maps are bijective for identity pass
- Resolved dtype on lowered tensors matches policy precedence
- View outputs share storage; `ALIAS` not double-charged (byte check in memory phase)
- `Maximum` identity produces `ALLOCATE` + correct FLOPs
- Saved-for-backward ports produce `SAVE` events (identity path)
- Invalid `resource_events()` (negative FLOPs) fails at lowering, not at graph build
- Implementation pin selects specific descriptor; conflicting pins raise
- ReLU pattern: `maximum(x, zero)` selects `maximum/relu-mask`
- ReLU mask: one `SAVE` on mask, not two operand saves; backward FLOPs = `numel` not `2×numel`
- Generic `maximum(a, b)` with non-zero `b` selects `maximum/identity`

---

## 16. Implementation order

1. **`lowered.py`** — records + maps + `ImplementationSelection`
2. **`InvocationContext`** + builder helper (defaults for FP32 reference backend)
3. **`LoweringRegistry`** + selection algorithm
4. **Identity registry** — `register_identity_defaults()` for all families
5. **`lower()`** — tensor materialization, estimation-context bridge, impl dispatch, SAVE synthesis
6. **`LoweredOperationValidator`**
7. **`tests/test_lowering.py`**
8. **`ReLUMaskImplementation`** + `register_maximum_implementations()`
9. Stub **`estimation.py`** (full `estimate()` when memory/FLOP land)

Steps 1–7 deliver a complete lowered graph ready for the memory and FLOP pipeline.

---

## 17. Anti-patterns

- **Subclassing `Operation` for ReLU/CUDA/fused kernels** — couples semantics to backend; breaks ADR-0002.
- **Putting impl choice in `Module.forward`** — selection belongs in `lower()`.
- **Auto-registering from `Operation.family` via introspection** — hides explicit registry; obscures rejection reasons and pins.
- **One global `implementation: str` on `Operation`** — structural declarations must stay backend-neutral.
- **Memory/FLOP accounting reading structural ops directly** — only consume `LoweredGraph`.

---

## 18. Summary

Lowering is a **new pass** that:

- takes a frozen structural graph plus an invocation context,
- selects implementations from an explicit registry,
- resolves accounting metadata,
- invokes estimation hooks that structural validation deliberately skips,
- emits a self-contained `LoweredGraph` with structural mappings and selection records.

Identity wrappers bootstrap all existing families immediately. Specialized
implementations (starting with `maximum/relu-mask`) add alternate event/FLOP
behavior without changing structural `Operation` classes. Everything downstream
(memory peak, total bytes, FLOP reports) reads `LoweredGraph` only.
