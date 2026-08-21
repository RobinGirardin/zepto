# Zepto VRAM Accounting Plan

## Goal

Implement an explicit, backend-neutral VRAM accounting model for Zepto.
Structural graph construction must describe model semantics without charging
memory or requiring hardware/runtime decisions. Lowering must then select a
concrete execution strategy for one invocation, and dedicated accounting APIs
must compute both total allocation and peak live allocation from the lowered
graph.

The resulting design must support:

- explicit tensor and parameter precision;
- default precision policies with explicit overrides;
- storage roles, persistence, and gradient participation;
- ordinary tensor representation for saved backward values;
- internal auxiliary tensors that are not public operation results;
- view aliases versus materialized copies;
- operation workspace and lowered-kernel temporaries;
- persistent state and optimizer state;
- total allocation (`sum-all`) and peak live allocation;
- optimizer-specific policies;
- focused validation that does not perform memory accounting during graph
  construction.

This plan does not introduce runtime execution. It defines the structural
contracts and accounting simulation needed before backend execution is added.

## Domain rules

The implementation must preserve these distinctions:

1. `ValueKind` is a port-contract category. It describes whether a port carries
   a tensor, parameter, state, or gradient.
2. `TensorMetadata.semantic_type` describes the mathematical or domain meaning
   of a tensor, such as `hidden_state`, `attention_scores`, or
   `auxiliary_state`.
3. Accounting metadata describes storage and lifetime behavior, such as dtype,
   role, persistence, and gradient participation.
4. `requires_grad` belongs to graph-value metadata. `trainable` belongs to
   parameters.
5. Saved backward values are ordinary tensors. Their saved lifetime is an
   explicit operation relationship, not an opaque memory record.
6. Public operation outputs and internal auxiliary tensors are both graph
   tensors, but only public outputs are returned by functional wrappers.
7. A view creates a new tensor identity sharing an existing `StorageId`.
8. A materialized copy creates a new tensor identity and a new `StorageId`.
9. Lowering selects implementations and emits execution-specific storage
   behavior. Structural graph construction does not perform memory accounting.
10. Resource events use ordinary allocation, release, alias, persistence, save,
    and workspace behavior. There is no dedicated gradient event.

## 1. Add typed accounting metadata

### Files

- `src/zepto/core/metadata.py`
- `src/zepto/core/parameter.py`
- `src/zepto/core/__init__.py`

### Code changes

Add a backend-neutral dtype declaration. The exact representation may be an
enum or immutable value object, but it must provide an unambiguous byte width.
The initial built-in values should include at least:

```python
class DType(StrEnum):
    UNKNOWN = "unknown"
    BOOL = "bool"
    INT32 = "int32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"

    @property
    def itemsize(self) -> int | None:
        ...
```

Add a storage/accounting role enum. Keep this separate from `ValueKind` and
from `semantic_type`:

```python
class TensorRole(StrEnum):
    INPUT = "input"
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    AUXILIARY = "auxiliary"
    STATE = "state"
    WORKSPACE = "workspace"
    GRADIENT = "gradient"
```

Extend `TensorMetadata`:

```python
@dataclass(frozen=True, slots=True)
class TensorMetadata:
    shape: Shape
    semantic_type: str = "tensor"
    dtype: DType | None = None
    requires_grad: bool = False
    role: TensorRole | None = None
    persistent: bool = False
```

Keep the existing concrete-dimension and non-empty-semantic-type checks.
Validate that `dtype`, when supplied, is a `DType`, that `role`, when
supplied, is a `TensorRole`, and that persistent metadata is explicit and
boolean.

Do not add `trainable` to `TensorMetadata`. Extend `Parameter` with an explicit
trainability field if it is not already equivalent to `requires_grad`:

```python
@dataclass(frozen=True, slots=True)
class Parameter:
    id: ParameterId
    metadata: TensorMetadata
    trainable: bool = True
```

If compatibility requires retaining the existing `requires_grad` constructor
argument, define it as parameter gradient participation and keep
`trainable` separate.

Update `metadata_compatible()` to compare all declaration fields that are
contractually significant. An unspecified dtype or role is a wildcard during
declaration matching and is resolved by the estimation context.

Export the new types from `zepto.core` and the package-level public API.

## 2. Add precision and accounting policy resolution

### Files

- `src/zepto/core/operation/records.py`
- new `src/zepto/core/accounting.py`
- `src/zepto/core/__init__.py`

### Code changes

Replace the free-form `EstimationContext.dtype: str` with typed context data.
The context must contain hardware/backend-independent defaults and policy
overrides:

```python
@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    default_dtype: DType
    by_role: tuple[tuple[TensorRole, DType], ...] = ()
    by_semantic_type: tuple[tuple[str, DType], ...] = ()

    def resolve(self, metadata: TensorMetadata) -> DType:
        ...
```

Define the precedence exactly as:

```text
explicit tensor/parameter dtype
    > operation or port override
    > semantic-type or role policy
    > context-wide default
```

The operation/port override may be represented as a resolved copy of
`TensorMetadata`, a dedicated port policy, or an equivalent immutable record.
It must not mutate the structural graph.

Add a resolver:

```python
class AccountingPolicy:
    precision: PrecisionPolicy

    def resolve_dtype(self, metadata: TensorMetadata) -> DType:
        ...

    def bytes_for(self, metadata: TensorMetadata) -> int:
        ...
```

`bytes_for()` must compute `numel(shape) * dtype.itemsize` and reject
unresolved `DType.UNKNOWN` when accounting requires a byte count.

`EstimationContext` must expose resolved metadata for named input, output,
auxiliary, parameter, and state ports. Keep context immutable.

## 3. Separate public output ports from auxiliary tensor ports

### Files

- `src/zepto/core/operation/base.py`
- `src/zepto/core/operation/records.py`
- `src/zepto/core/operation/structural.py`
- `src/zepto/core/ports.py`
- `src/zepto/core/graph.py`
- `src/zepto/core/composition.py`

### Code changes

Add an auxiliary-port declaration to the operation contract:

```python
class SemanticOperation(ABC):
    @property
    @abstractmethod
    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        """Return internal tensor ports used by forward/backward behavior."""
        ...
```

The default implementation for operations without auxiliary values should be
an empty tuple, so existing primitive operations remain simple.

Auxiliary ports must:

- use tensor-compatible `ValueKind`;
- have names unique across input, public output, and auxiliary ports;
- have concrete metadata after inference;
- be represented by graph `Tensor` objects;
- receive their own tensor and storage identities;
- be eligible for `saved_for_backward` references;
- not be returned as public functional outputs.

Extend `OperationResult`:

```python
@dataclass(frozen=True, slots=True)
class OperationResult:
    outputs: tuple[TensorMetadata, ...]
    auxiliary_outputs: tuple[TensorMetadata, ...] = ()
    aliases: tuple[AliasSpec | None, ...] = ()
    auxiliary_aliases: tuple[AliasSpec | None, ...] = ()
    saved_for_backward: tuple[str, ...] = ()
```

The result must validate arity separately for public outputs and auxiliary
outputs. Auxiliary output names come from `auxiliary_ports`.

Update `StructuralOperation` to store:

- public output ports and tensor IDs;
- auxiliary output ports and tensor IDs;
- the same `OperationResult`;
- saved references that may resolve to input, public output, or auxiliary
  output ports.

Update `StructuralGraphBuilder.add_operation()` to:

1. accept auxiliary ports and auxiliary metadata;
2. validate all port names and value kinds together;
3. allocate graph tensor IDs for public and auxiliary outputs;
4. assign storage according to the corresponding alias declaration;
5. return only public output tensor IDs to composition code;
6. retain auxiliary tensor IDs in the structural operation.

Update `GraphCompositionContext.apply()` so functional wrappers continue to
return only public outputs. Operations such as normalization can retain
internal values without exposing them in the module API.

## 4. Represent saved backward values explicitly

### Files

- `src/zepto/core/operation/records.py`
- `src/zepto/core/operation/base.py`
- `src/zepto/core/graph.py`
- primitive operation implementations that save state, especially ReLU and
  the arithmetic operations used by normalization modules

### Code changes

Keep `BackwardSpec.saved_for_backward` as named port references. Extend its
validation domain from input/public-output ports to all input, public-output,
and auxiliary ports.

For a saved value produced by primitive composition:

```python
class Divide(Operation):
    ...

    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        names: list[str] = []
        if inputs[0].require_grad:
            names.append("right")
        if inputs[1].require_grad:
            names.append("left")
        return tuple(names)
```

`BackwardSpec.saved_for_backward` declares the names a primitive is allowed to
save. The operation-specific `saved_for_backward(inputs, outputs)` method
selects the subset needed for this concrete invocation. The selection takes
the inferred output metadata rather than a constructed `OperationResult`, so
the base `Operation.infer_result()` orchestrator can compose the
`OperationResult` exactly once from the `infer_outputs()`, `output_aliases()`,
and `saved_for_backward()` hooks. This keeps the
dependency decision in the lowest-level operation that owns the derivative,
rather than in a composite module.

For example, `RMSNorm(Module)` composes primitive operations that compute
`rstd`, then passes that ordinary graph tensor into `Multiply(Operation)`.
`Multiply` saves `rstd` when the other operand requires its gradient. The
`RMSNorm` module does not declare itself as an operation and does not own the
saved-state contract.

The structural graph must resolve every selected saved name to a concrete
`PortRef` and retain that reference in `StructuralOperation.saved_for_backward`.

Do not encode saved values as opaque records or infer their lifetime solely
from `semantic_type`. The saved relationship is explicit; tensor metadata
controls dtype, role, persistence, and gradient participation.

## 5. Correct alias and materialization semantics

### Files

- `src/zepto/core/operation/records.py`
- `src/zepto/core/operation/base.py`
- `src/zepto/core/graph.py`
- new or updated resource-event tests

### Code changes

Retain:

```python
class Materialization(StrEnum):
    VIEW = "view"
    CONTIGUOUS_COPY = "contiguous_copy"
```

Update result validation:

- `VIEW` must refer to a valid input port;
- `VIEW` must preserve element count and satisfy the operation's view
  compatibility rules;
- `CONTIGUOUS_COPY` must refer to a valid input port;
- both modes must have explicit, known semantics.

Fix `StructuralGraphBuilder.add_operation()`. It currently reuses the source
storage for every non-`None` alias. Change it so that:

```python
if alias is not None and alias.materialization is Materialization.VIEW:
    storage_id = source_tensor.storage_id
else:
    storage_id = allocate_new_storage_id()
```

`CONTIGUOUS_COPY` therefore receives a fresh storage identity even though the
output is semantically derived from an input.

Resource-event generation must distinguish the two cases:

- a view emits an alias event and no output allocation event;
- a contiguous copy emits an allocation event for its new storage.

Update `Identity`, `Reshape`, and `Transpose` to continue declaring views.
Add a test operation that declares `CONTIGUOUS_COPY`.

## 6. Decompose validation into dedicated validators

### Files

- new `src/zepto/core/operation/validation.py`
- `src/zepto/core/operation/base.py`
- new `src/zepto/core/graph_validation.py` or equivalent graph module
- `src/zepto/core/graph.py`

### Code changes

Create focused validators with clear responsibilities:

```python
class DeclarationValidator:
    def validate(self, operation: Operation) -> None: ...
    def validate_family(self, operation: Operation) -> None: ...
    def validate_ports(self, operation: Operation) -> None: ...
    def validate_backward(self, operation: Operation) -> None: ...


class InvocationValidator:
    def validate_inputs(
        self,
        operation: Operation,
        inputs: tuple[TensorMetadata, ...],
    ) -> None: ...

    def validate_result(
        self,
        operation: Operation,
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None: ...

    def validate_aliases(self, ...) -> None: ...
    def validate_saved_state(self, ...) -> None: ...
    def validate_metadata(self, ...) -> None: ...


class GraphValidator:
    def validate(self, graph: StructuralGraph) -> None: ...
    def validate_ownership(self, graph: StructuralGraph) -> None: ...
    def validate_operation_order(self, graph: StructuralGraph) -> None: ...
    def validate_port_bindings(self, graph: StructuralGraph) -> None: ...
    def validate_storage_links(self, graph: StructuralGraph) -> None: ...
    def validate_graph_outputs(self, graph: StructuralGraph) -> None: ...
```

Keep `Operation.validate_declaration()`, `Operation.infer()`,
`Operation.validate_result()`, and `StructuralGraph.validate()` as small
public orchestration methods delegating to these validators.

Remove resource-event invocation from `Operation.validate_result()`. Result
validation may check static event declarations or event record shape only when
an event sequence is explicitly requested. It must not call
`resource_events()` during structural graph construction.

FLOP validation should likewise happen when FLOPs are accounted for in a
lowered invocation, not by fabricating an incomplete execution context while
validating a structural result.

## 7. Remove the dedicated gradient resource event

### Files

- `src/zepto/core/operation/records.py`
- `src/zepto/core/operation/__init__.py`
- `src/zepto/core/__init__.py`
- tests and documentation referring to `ResourceEventKind.GRADIENT`

### Code changes

Remove `ResourceEventKind.GRADIENT`.

Represent gradient memory through ordinary tensor and port metadata:

- gradient ports use `ValueKind.GRADIENT` where appropriate;
- gradient tensors use `TensorRole.GRADIENT`;
- allocation uses `ALLOCATE`;
- accumulation/retention uses `PERSIST`;
- release uses `RELEASE`.

Add tests proving that gradient accumulation can be represented without a
gradient-specific event kind.

## 8. Introduce lowered execution records

### Files

- new `src/zepto/core/lowering.py`
- new `src/zepto/core/lowered.py`
- new `src/zepto/core/estimation.py`
- `src/zepto/core/__init__.py`

### Code changes

Define an immutable invocation context:

```python
@dataclass(frozen=True, slots=True)
class InvocationContext:
    phase: str
    hardware: str
    backend: str
    precision: PrecisionPolicy
    state: tuple[tuple[str, object], ...] = ()
    policies: tuple[object, ...] = ()
```

Define lowered records that preserve mappings back to the structural graph:

```python
@dataclass(frozen=True, slots=True)
class LoweredTensor:
    id: str
    structural_tensor_id: TensorId | None
    metadata: TensorMetadata
    storage_id: str
    workspace: bool = False


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    id: str
    structural_operation_id: OperationId
    implementation: str
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    auxiliary_tensors: tuple[str, ...]
    resource_events: tuple[ResourceEvent, ...]
    forward_flops: int
    backward_flops: int


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    tensors: Mapping[str, LoweredTensor]
    operations: tuple[LoweredOperation, ...]
    context: InvocationContext
```

Define:

```python
def lower(
    graph: StructuralGraph,
    context: InvocationContext,
) -> LoweredGraph:
    ...
```

`lower()` resolves concrete dtype, implementation, storage behavior,
workspace, and resource events for one invocation. It does not mutate the
structural graph.

The lowering record must retain structural-to-lowered tensor and operation
mappings for reporting and reproducibility.

## 9. Implement memory accounting

### Files

- new `src/zepto/core/memory.py`
- `src/zepto/core/estimation.py`
- `src/zepto/core/__init__.py`

### Code changes

Define a memory report:

```python
@dataclass(frozen=True, slots=True)
class MemoryReport:
    total_allocation_bytes: int
    peak_live_bytes: int
    persistent_bytes: int
    workspace_peak_bytes: int
    saved_tensor_peak_bytes: int
    by_storage: Mapping[str, int]
    by_operation: Mapping[str, int]
    timeline: tuple["MemorySnapshot", ...]
```

Define accounting policy and snapshots:

```python
@dataclass(frozen=True, slots=True)
class MemoryAccountingPolicy:
    include_parameters: bool = True
    include_workspace: bool = True
    include_saved_tensors: bool = True
    include_persistent_storage: bool = True


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    event_index: int
    live_bytes: int
    allocated_bytes: int
    active_storage: tuple[str, ...]
```

Implement:

```python
def account_memory(
    lowered: LoweredGraph,
    policy: MemoryAccountingPolicy | None = None,
) -> MemoryReport:
    ...
```

The simulator must:

1. resolve each tensor's byte size from concrete metadata;
2. process lowered resource events in order;
3. allocate each storage identity at most once per lifetime;
4. treat aliases/views as references to existing storage;
5. track releases by storage identity, not only tensor identity;
6. retain persistent storage until its policy-defined release;
7. include workspace storage separately when configured;
8. compute `total_allocation_bytes` as the sum of allocation events;
9. compute `peak_live_bytes` as the maximum live storage total;
10. record attribution by operation and storage;
11. reject negative live memory and invalid release/alias sequences.

Use storage identity as the primary accounting key. Multiple tensor identities
sharing one view storage must not be charged multiple times.

## 10. Implement FLOP accounting and the convenience estimator

### Files

- new `src/zepto/core/flops.py`
- new or updated `src/zepto/core/estimation.py`
- `src/zepto/core/__init__.py`

### Code changes

Define:

```python
@dataclass(frozen=True, slots=True)
class FlopReport:
    forward_flops: int
    backward_flops: int
    total_flops: int
    by_operation: Mapping[str, int]
```

Implement:

```python
def account_flops(lowered: LoweredGraph) -> FlopReport:
    ...
```

Validate non-negative integer FLOP counts while accounting the lowered graph.
Preserve Zepto's convention that one multiply-add counts as two FLOPs.

Define a convenience result:

```python
@dataclass(frozen=True, slots=True)
class CostReport:
    memory: MemoryReport
    flops: FlopReport
    lowered: LoweredGraph
```

Implement:

```python
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

`estimate()` is a convenience composition API. `lower()`,
`account_memory()`, and `account_flops()` remain independently usable.

## 11. Add optimizer policy contracts

### Files

- new `src/zepto/core/optimizer.py`
- `src/zepto/core/parameter.py`
- `src/zepto/core/lowering.py`
- `src/zepto/core/memory.py`
- `src/zepto/core/__init__.py`

### Code changes

Define an optimizer-policy interface:

```python
class OptimizerPolicy(ABC):
    @abstractmethod
    def state_metadata(
        self,
        parameter: Parameter,
        context: InvocationContext,
    ) -> tuple[TensorMetadata, ...]:
        ...

    @abstractmethod
    def update_events(
        self,
        parameter: Parameter,
        context: InvocationContext,
    ) -> tuple[ResourceEvent, ...]:
        ...
```

The policy must describe optimizer-state tensors, their dtype, persistence,
and update-time workspace/resource behavior. It must not be embedded in
operation declarations.

Add an initial simple implementation for SGD and a stateful implementation
for Adam only if the current scope requires examples. The interface must
support future optimizers without changing the memory analyzer.

## 12. Update operation implementations

### Files

- `src/zepto/core/operation/*.py`
- `src/zepto/core/functional/*.py`
- `src/zepto/modules/*.py`

### Code changes

Update every primitive operation to implement the new metadata contract and
the standardized saved-value dependency contract. `BackwardSpec` declares
possible saved ports; `saved_for_backward(inputs, result)` selects the
concrete subset for the invocation.

Required behavior:

- `Identity`, `Reshape`, and `Transpose` declare view aliases and zero FLOPs.
- Ordinary arithmetic and matrix operations allocate fresh output storage.
- `CONTIGUOUS_COPY` operations allocate fresh storage.
- Add `Divide`, `SquareRoot`, and `Subtract` operation declarations together
  with their functional wrappers. Use the canonical spelling `Subtract`.
- `Add`, `Identity`, `Reshape`, `Transpose`, and `Split` declare no saved
  forward values unless their derivative contract later requires one.
- `Multiply` conditionally saves the opposite operand for each requested
  gradient.
- `MatMul` conditionally saves only the operands required by the requested
  gradients.
- `ReLU` saves its output, which is an ordinary graph tensor used to recover
  the backward mask.
- `Divide` conditionally saves the denominator when the numerator gradient is
  needed and the numerator when the denominator gradient is needed.
- `SquareRoot` saves its input or output according to the selected derivative
  formulation.
- `Subtract` declares the ordinary subtraction backward dependencies.
- `LayerNorm` and `RMSNorm` remain `Module` compositions. They derive their
  behavior from primitive wrappers such as `Subtract`, `Multiply`, `Divide`,
  and `SquareRoot`; they do not implement operation-level saved-state logic.
- Intermediate primitive outputs such as `rstd` are ordinary graph tensors.
  They may be consumed inside a module without being returned by that module.
- operation resource-event methods describe lowered behavior, not graph
  construction behavior.

Functional wrappers remain thin and return only public output tensors.

## 13. Add tests

### Files

- extend `tests/test_operation_contract.py`
- extend `tests/test_structural_graph.py`
- add `tests/test_metadata_accounting.py`
- add `tests/test_memory_accounting.py`
- add `tests/test_lowering.py`
- add `tests/test_optimizer_policy.py`

### Required cases

Test that:

- dtype, role, persistence, and gradient metadata validate correctly;
- explicit dtype overrides policy defaults;
- role and semantic-type defaults apply when dtype is unspecified;
- trainability is parameter-specific;
- auxiliary outputs are graph tensors but are not public wrapper results;
- saved backward references can resolve auxiliary ports;
- each primitive's `saved_for_backward(inputs, result)` selects only the
  dependencies required by the concrete gradient inputs;
- `Divide`, `SquareRoot`, and `Subtract` expose correct semantic, FLOP,
  resource, and backward contracts;
- `LayerNorm` and `RMSNorm` are `Module` compositions and do not appear as
  primitive operation declarations;
- normalization intermediates such as `rstd` are ordinary graph tensors
  produced and consumed by primitive operations;
- unknown saved ports fail validation;
- view aliases share storage and are charged once;
- contiguous copies receive distinct storage and allocation charges;
- gradient memory uses ordinary allocation/persistence/release events;
- resource events are not generated during structural graph construction;
- declaration, invocation, and graph validators report focused failures;
- lowering preserves structural mappings;
- workspace contributes to memory only when policy enables it;
- persistent tensors survive until their release;
- `sum-all` and peak memory differ correctly for overlapping lifetimes;
- aliasing does not inflate either metric;
- invalid event sequences are rejected;
- FLOP accounting preserves forward/backward attribution;
- `estimate()` composes lowering, memory accounting, and FLOP accounting;
- optimizer policies expose state tensors and update resource events.

## 14. Update documentation and exports

### Files

- `CONTEXT.md`
- `docs/adr/0006-tensor-accounting-and-memory-reports.md`
- `docs/adr/0003-resource-events-and-horizon-simulation.md`
- `src/zepto/core/__init__.py`
- `src/zepto/__init__.py`

### Code/documentation changes

Document:

- the distinction between `ValueKind`, `semantic_type`, and accounting
  metadata;
- auxiliary tensors and public outputs;
- view versus materialized-copy storage;
- the absence of a dedicated gradient resource event;
- the lowering/accounting API boundary;
- `MemoryReport` total and peak metrics;
- precision override precedence;
- optimizer policies.

Export the public APIs:

```python
DType
TensorRole
PrecisionPolicy
AccountingPolicy
InvocationContext
LoweredGraph
lower
MemoryAccountingPolicy
MemoryReport
account_memory
FlopReport
account_flops
CostReport
estimate
OptimizerPolicy
```

## Completion criteria

The work is complete when:

1. Structural graph construction remains backend-neutral and performs no
   invocation-specific memory accounting.
2. Every graph tensor has enough metadata to resolve byte size under a
   context policy.
3. Auxiliary saved values are ordinary graph tensors with explicit saved-state
   references.
4. Views and materialized copies produce correct storage identities.
5. Lowering emits an immutable invocation-specific graph and mapping record.
6. `account_memory()` returns both total allocation and peak live allocation.
7. `account_flops()` and `estimate()` provide the documented public API.
8. Optimizer policies can contribute persistent state and update workspace.
9. Validation is decomposed into focused validators.
10. Focused tests and the complete test suite pass.

