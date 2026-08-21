# VRAM Accounting — Immediate Next Steps Implementation Plan

Derived from `operation-vram-plan.md` and the current state of the codebase.

## Current state assessment

Already in place (from the operation-contract migration):

- `Operation` contract with `infer_outputs()`, `output_aliases()`, `backward`,
  and the invocation-specific `saved_for_backward(inputs, outputs)` hook.
- `OperationResult(outputs, aliases, saved_for_backward)` composed exactly
  once by `Operation.infer_result()`.
- `Materialization` enum (`VIEW`, `CONTIGUOUS_COPY`) and `AliasSpec`.
- `StructuralOperation.saved_for_backward` resolved to `PortRef`s by
  `StructuralGraphBuilder.add_operation()`.
- Primitives: `Add`, `Divide`, `Identity`, `MatMul`, `Maximum`, `Minimum`,
  `Multiply`, `Reshape`, `Split`, `SquareRoot`, `Substract`, `Transpose`,
  with functional wrappers.

Not yet started, and unblocked (no dependency on later workstreams):

| Plan § | Item | Status |
|--------|------|--------|
| 1 | `DType`, `TensorRole`, extended `TensorMetadata`, `Parameter.trainable` | Not started |
| 2 | `PrecisionPolicy`, `AccountingPolicy`, typed `EstimationContext` | Not started (`EstimationContext.dtype` is still a free-form `str`) |
| 5 | VIEW vs `CONTIGUOUS_COPY` storage in the graph builder | Enum exists; builder reuses source storage for **every** alias |
| 6 | Validator decomposition; stop calling `resource_events()` during structural validation | Not started (`Operation.validate_result()` still calls `resource_events()` and both FLOP hooks) |
| 7 | Remove `ResourceEventKind.GRADIENT` | Not started |
| 12 (part) | `Subtract` canonical spelling | `Substract` misspelling everywhere |

Known latent bug found during assessment: `StructuralGraphBuilder.add_parameter()`
calls `Parameter(parameter_id, metadata, requires_grad)` with three arguments,
but `Parameter` only declares `id` and `metadata`. Step 1 fixes this.

Deferred (blocked on the steps above): auxiliary ports (§3), lowering (§8),
memory accounting (§9), FLOP accounting / `estimate()` (§10), optimizer
policies (§11), real `LayerNorm`/`RMSNorm` compositions (§12).

Execution order: Step 1 → Step 2, and independently Step 3 → Step 4 → Step 5;
Step 6 (naming) can run in parallel with any of them. Step 7 (tests)
accompanies each step.

---

## Step 1 — Typed accounting metadata (plan §1)

### Files

- `src/zepto/core/metadata.py` (edit)
- `src/zepto/core/parameter.py` (edit)
- `src/zepto/core/graph.py` (fix `add_parameter`)
- `src/zepto/core/__init__.py`, `src/zepto/__init__.py` (exports)

### 1.1 `DType` enum — `metadata.py`

```python
class DType(StrEnum):
    """Backend-neutral element type with an unambiguous byte width."""

    UNKNOWN = "unknown"
    BOOL = "bool"
    INT32 = "int32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"

    @property
    def itemsize(self) -> int | None:
        """Return the element byte width, or None for UNKNOWN."""
        return _DTYPE_ITEMSIZE[self]


_DTYPE_ITEMSIZE: dict[DType, int | None] = {
    DType.UNKNOWN: None,
    DType.BOOL: 1,
    DType.INT32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP32: 4,
    DType.FP64: 8,
}
```

### 1.2 `TensorRole` enum — `metadata.py`

```python
class TensorRole(StrEnum):
    """Storage/accounting role. Distinct from ValueKind and semantic_type."""

    INPUT = "input"
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    AUXILIARY = "auxiliary"
    STATE = "state"
    WORKSPACE = "workspace"
    GRADIENT = "gradient"
```

### 1.3 Extend `TensorMetadata` — `metadata.py`

Rename `require_grad` → `requires_grad` (canonical plan spelling) and add the
accounting fields:

```python
@dataclass(frozen=True, slots=True)
class TensorMetadata:
    shape: Shape
    semantic_type: str = "tensor"
    requires_grad: bool = False
    dtype: DType | None = None
    role: TensorRole | None = None
    persistent: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("Tensor dimensions must be non-negative integers")
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")
        if self.dtype is not None and not isinstance(self.dtype, DType):
            raise ValueError("dtype must be a DType when supplied")
        if self.role is not None and not isinstance(self.role, TensorRole):
            raise ValueError("role must be a TensorRole when supplied")
        if not isinstance(self.persistent, bool):
            raise ValueError("persistent must be an explicit boolean")
        if not isinstance(self.requires_grad, bool):
            raise ValueError("requires_grad must be an explicit boolean")
```

`require_grad` → `requires_grad` rename ripples through:
`operation/{add,divide,matmul,maximum,minimum,multiply,reshape,split,square_root,substract,transpose}.py`,
`operation/helpers.py`, `tests/test_operation_contract.py` (22 uses). Pure
mechanical rename; do it in the same commit.

### 1.4 Update `metadata_compatible()` — `metadata.py`

Unspecified `dtype`/`role` on the expected side is a wildcard; a specified
value must match exactly. `persistent` and `requires_grad` are not part of
declaration matching (they are value properties, not port contracts):

```python
def metadata_compatible(expected: TensorMetadata, actual: TensorMetadata) -> bool:
    if expected.semantic_type != actual.semantic_type:
        return False
    if expected.shape != actual.shape:
        return False
    if expected.dtype is not None and expected.dtype != actual.dtype:
        return False
    if expected.role is not None and expected.role != actual.role:
        return False
    return True
```

### 1.5 `Parameter.trainable` — `parameter.py`

```python
@dataclass(frozen=True, slots=True)
class Parameter:
    id: ParameterId
    metadata: TensorMetadata
    trainable: bool = True
```

`trainable` means "the optimizer updates this parameter"; gradient
participation stays on `metadata.requires_grad`. Fix
`StructuralGraphBuilder.add_parameter()` accordingly:

```python
def add_parameter(
    self,
    metadata: TensorMetadata,
    *,
    trainable: bool = True,
) -> ParameterId:
    ...
    self._parameters[parameter_id] = Parameter(parameter_id, metadata, trainable)
```

Callers that passed `requires_grad=` to `add_parameter` must now set it on
the metadata and pass `trainable=` for optimizer participation.

### 1.6 Exports

Add `DType` and `TensorRole` to `zepto.core.__all__` and re-export from
`zepto/__init__.py`. Add `Parameter` to `__all__` (currently imported but not
exported).

---

## Step 2 — Precision and accounting policy resolution (plan §2)

**Design split.** Operations emit `ResourceEvent` records that describe *what*
happens at each step (allocate, release, alias, save, persist, workspace).
`PrecisionPolicy` and `AccountingPolicy` sit on the evaluation side: given an
event and the tensor metadata it refers to, they resolve dtype and compute how
many bytes are allocated, released, or retained. Structural graph construction
and operation declarations produce events; accounting policies quantify them.

### Files

- new `src/zepto/core/accounting.py`
- `src/zepto/core/operation/records.py` (retype `EstimationContext`)
- `src/zepto/core/__init__.py`

### 2.1 `PrecisionPolicy` — `accounting.py`

```python
@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Default dtypes with role- and semantic-type-specific overrides."""

    default_dtype: DType
    by_role: tuple[tuple[TensorRole, DType], ...] = ()
    by_semantic_type: tuple[tuple[str, DType], ...] = ()

    def __post_init__(self) -> None:
        if self.default_dtype is DType.UNKNOWN:
            raise ValueError("Precision policy default cannot be UNKNOWN")
        # reject duplicate keys in by_role / by_semantic_type

    def resolve(self, metadata: TensorMetadata) -> DType:
        """Resolve precedence: explicit dtype > semantic-type > role > default."""
        if metadata.dtype is not None and metadata.dtype is not DType.UNKNOWN:
            return metadata.dtype
        for semantic_type, dtype in self.by_semantic_type:
            if semantic_type == metadata.semantic_type:
                return dtype
        if metadata.role is not None:
            for role, dtype in self.by_role:
                if role == metadata.role:
                    return dtype
        return self.default_dtype
```

Precedence (plan §2): explicit tensor/parameter dtype > operation/port
override > semantic-type or role policy > context-wide default. The
operation/port override slot is realized by passing a resolved copy of
`TensorMetadata` (via `dataclasses.replace(metadata, dtype=...)`) into the
context — never by mutating the structural graph.

### 2.2 `AccountingPolicy` — `accounting.py`

```python
def numel(shape: Shape) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


@dataclass(frozen=True, slots=True)
class AccountingPolicy:
    precision: PrecisionPolicy

    def resolve_dtype(self, metadata: TensorMetadata) -> DType:
        return self.precision.resolve(metadata)

    def bytes_for(self, metadata: TensorMetadata) -> int:
        dtype = self.resolve_dtype(metadata)
        itemsize = dtype.itemsize
        if itemsize is None:
            raise ValueError(
                f"Cannot account bytes for unresolved dtype {dtype!r}"
            )
        return numel(metadata.shape) * itemsize
```

### 2.3 Retype `EstimationContext` — `operation/records.py`

Remove the free-form `dtype: str` and `layout: str` fields. The context keeps
resolved per-port metadata (already present as `port_metadata`) and gains an
optional precision policy so estimators can resolve byte sizes:

```python
@dataclass(frozen=True, slots=True)
class EstimationContext:
    phase: str = "forward"
    port_metadata: tuple[tuple[str, TensorMetadata], ...] = ()
    precision: PrecisionPolicy | None = None
    state: tuple[tuple[str, Any], ...] = ()

    def metadata_for(self, port_name: str) -> TensorMetadata | None: ...
```

`port_metadata` entries are the resolved copies for named input, output,
auxiliary (future §3), parameter, and state ports. Context stays immutable.
No current estimator reads `context.dtype`/`context.layout` (verify with a
grep before deleting), so the removal is safe.

Import note: `records.py` must not import `accounting.py` if that creates a
cycle — `PrecisionPolicy` only depends on `metadata.py`, so either keep
`PrecisionPolicy` in `accounting.py` and import it in `records.py`
(accounting → metadata, records → accounting, no cycle), or move
`PrecisionPolicy` into `metadata.py` and keep only `AccountingPolicy` in
`accounting.py`. Prefer the first; fall back to the second if imports fight.

### 2.4 Exports

Export `PrecisionPolicy`, `AccountingPolicy` from `zepto.core`.

---

## Step 3 — Correct alias and materialization storage (plan §5)

### Files

- `src/zepto/core/graph.py` (edit `add_operation`)
- `src/zepto/core/operation/base.py` (alias validation, small edit)
- `tests/test_structural_graph.py`, `tests/test_operation_contract.py`

### 3.1 Fix `StructuralGraphBuilder.add_operation()`

The output-storage loop currently reuses the source tensor's storage for
every non-`None` alias. Change the alias branch to check materialization:

```python
alias = result.aliases[index] if result.aliases else None
storage_id = None
if alias is not None and alias.materialization is Materialization.VIEW:
    source_index = next(
        position
        for position, input_port in enumerate(input_ports)
        if input_port.name == alias.source_port
    )
    storage_id = self._tensors[input_tensors[source_index]].storage_id
if storage_id is None:
    storage_id = StorageId(self._graph_id, self._next_storage)
    self._next_storage += 1
```

A `CONTIGUOUS_COPY` alias thus records the semantic derivation (via
`AliasSpec.source_port` in the result) but receives a fresh `StorageId`.
Import `Materialization` into `graph.py`.

### 3.2 Tighten alias validation in `Operation.validate_result()`

Both modes already require a valid input source port. Keep the existing
element-count check for `VIEW` (`_numel(source) == _numel(output)`), and make
the materialization check explicit rather than comparing `.value == "view"`:

```python
if alias.materialization is Materialization.VIEW:
    ...  # element-count preservation check (existing)
elif alias.materialization is not Materialization.CONTIGUOUS_COPY:
    raise OperationError(f"Unknown materialization {alias.materialization!r}")
```

(When Step 4 lands, this logic moves into `InvocationValidator.validate_aliases`.)

### 3.3 Test operation with `CONTIGUOUS_COPY`

Add a test-local operation (in `tests/test_structural_graph.py`, not shipped)
that declares a copy:

```python
class Contiguous(Operation):
    @property
    def family(self) -> str:
        return "contiguous"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=True, gradient_inputs=("output",),
                            gradient_outputs=("input",))

    def infer_outputs(self, inputs):
        return (inputs[0],)

    def output_aliases(self):
        return (AliasSpec("input", Materialization.CONTIGUOUS_COPY),)

    def saved_for_backward(self, inputs, outputs):
        return ()

    def forward_flops(self, context):
        return 0

    def resource_events(self, context, result):
        ...
```

### 3.4 Required test cases

- `Identity`/`Reshape`/`Transpose` outputs share the input's `StorageId`
  (view charged once).
- The `Contiguous` test operation's output has a **distinct** `StorageId`
  from its input.
- A view whose element count differs from its source is rejected.

---

## Step 4 — Decompose validation into dedicated validators (plan §6)

### Files

- new `src/zepto/core/operation/validation.py`
- `src/zepto/core/operation/base.py` (slim down to orchestration)
- new `src/zepto/core/graph_validation.py`
- `src/zepto/core/graph.py` (delegate `StructuralGraph.validate()`)
- `src/zepto/core/operation/__init__.py`, `src/zepto/core/__init__.py`

### 4.1 `DeclarationValidator` — `operation/validation.py`

Move the body of the current `Operation.validate_declaration()` (base.py
lines 116–146) into focused methods:

```python
class DeclarationValidator:
    """Validate an operation's static declaration."""

    def validate(self, operation: "Operation") -> None:
        self.validate_family(operation)
        self.validate_ports(operation)
        self.validate_backward(operation)

    def validate_family(self, operation: "Operation") -> None:
        # non-empty str family

    def validate_ports(self, operation: "Operation") -> None:
        # tuples of named PortSpec; unique names across input+output
        # (extended to auxiliary ports in §3 later)

    def validate_backward(self, operation: "Operation") -> None:
        # BackwardSpec type; saved/gradient names reference known ports;
        # unsupported backward declares no requirements
```

### 4.2 `InvocationValidator` — `operation/validation.py`

Move the body of the current `Operation.validate_result()` (base.py lines
174–250), split by concern, and **delete the resource-event and FLOP
invocations** (lines 228–250):

```python
class InvocationValidator:
    """Validate one concrete invocation's inputs and inferred result."""

    def validate_inputs(
        self,
        operation: "Operation",
        inputs: tuple[TensorMetadata, ...],
    ) -> None:
        # arity matches input_ports; every value is TensorMetadata

    def validate_result(
        self,
        operation: "Operation",
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None:
        self.validate_metadata(operation, result)
        self.validate_aliases(operation, inputs, result)
        self.validate_saved_state(operation, inputs, result)

    def validate_metadata(self, operation, result) -> None:
        # OperationResult type; output arity vs output_ports;
        # all outputs are TensorMetadata

    def validate_aliases(self, operation, inputs, result) -> None:
        # alias arity; known source port; VIEW element-count preservation;
        # known Materialization member

    def validate_saved_state(self, operation, inputs, result) -> None:
        # selection is a unique tuple of non-empty names;
        # unsupported backward selects nothing;
        # result.saved_for_backward == operation.saved_for_backward(...);
        # subset of BackwardSpec.saved_for_backward;
        # subset of declared port names
```

Resource-event and FLOP validation move to lowering (§8, deferred): result
validation must not fabricate an `EstimationContext` or call
`resource_events()` during structural graph construction. If an event
sequence is explicitly requested (a future lowered invocation), its record
shape is checked there.

### 4.3 Slim `Operation` orchestration — `base.py`

```python
_DECLARATION_VALIDATOR = DeclarationValidator()
_INVOCATION_VALIDATOR = InvocationValidator()


class Operation(SemanticOperation, EstimationOperation, ABC):
    def validate_declaration(self) -> None:
        _DECLARATION_VALIDATOR.validate(self)

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        self.validate_declaration()
        _INVOCATION_VALIDATOR.validate_inputs(self, inputs)
        outputs = self.infer_outputs(inputs)
        if not isinstance(outputs, tuple):
            raise OperationError("infer_outputs must return a tuple")
        result = OperationResult(
            outputs=outputs,
            aliases=self.output_aliases(),
            saved_for_backward=self.saved_for_backward(inputs, outputs),
        )
        self.validate_result(inputs, result)
        return result

    def validate_result(self, inputs, result) -> None:
        _INVOCATION_VALIDATOR.validate_result(self, inputs, result)
```

Public API (`validate_declaration`, `infer_result`, `validate_result`) is
unchanged, so callers in `graph.py` and `composition.py` keep working.

### 4.4 `GraphValidator` — `graph_validation.py`

Move the body of `StructuralGraph.validate()` (graph.py lines 107–154):

```python
class GraphValidator:
    def validate(self, graph: "StructuralGraph") -> None:
        self.validate_ownership(graph)
        self.validate_port_bindings(graph)
        self.validate_operation_order(graph)
        self.validate_storage_links(graph)
        self.validate_graph_outputs(graph)

    def validate_ownership(self, graph) -> None:
        # tensors/operations belong to graph.id (CrossGraphReferenceError)

    def validate_port_bindings(self, graph) -> None:
        # declaration present and valid; result re-validated against input
        # metadata; saved PortRefs resolve; input/output arity (PortArityError)

    def validate_operation_order(self, graph) -> None:
        # producers precede consumers (InvalidOperationOrderError)

    def validate_storage_links(self, graph) -> None:
        # NEW: every view alias's output tensor shares its source tensor's
        # StorageId; every non-view output has a StorageId not shared with
        # any of the operation's input tensors

    def validate_graph_outputs(self, graph) -> None:
        # outputs registered in graph.tensors (InvalidGraphOutputError)


class StructuralGraph:
    def validate(self) -> None:
        GraphValidator().validate(self)
```

Note `graph_validation.py` imports from `graph.py` only under
`typing.TYPE_CHECKING` (or takes the graph structurally) to avoid a cycle;
`graph.py` imports `GraphValidator` normally.

### 4.5 Exports

Export `DeclarationValidator`, `InvocationValidator` from `zepto.core.operation`
and `GraphValidator` from `zepto.core`.

---

## Step 5 — Remove the dedicated gradient resource event (plan §7)

### Files

- `src/zepto/core/operation/records.py`
- `tests/test_operation_contract.py` (add coverage)

### Changes

Delete the `GRADIENT` member from `ResourceEventKind`:

```python
class ResourceEventKind(StrEnum):
    ALLOCATE = "allocate"
    RELEASE = "release"
    ALIAS = "alias"
    SAVE = "save"
    PERSIST = "persist"
    WORKSPACE = "workspace"
```

Grep confirms no operation, helper, or test currently emits
`ResourceEventKind.GRADIENT` (only the enum definition and docs mention it),
so this is a pure deletion. `ValueKind.GRADIENT` in `ports.py` **stays** —
gradient ports remain a port-contract category.

Gradient memory is henceforth represented with ordinary events plus metadata:

- gradient ports → `ValueKind.GRADIENT`;
- gradient tensors → `TensorRole.GRADIENT` (from Step 1);
- allocate → `ALLOCATE`, accumulate/retain → `PERSIST`, free → `RELEASE`.

Add a test constructing a gradient-accumulation event sequence
(`ALLOCATE` → `PERSIST` → `RELEASE` on a tensor whose metadata has
`role=TensorRole.GRADIENT`) proving no gradient-specific kind is needed.

---

## Step 6 — Naming alignment (plan §12, partial)

### Files

- rename `src/zepto/core/operation/substract.py` → `subtract.py`
- `src/zepto/core/operation/__init__.py`
- `src/zepto/core/functional/elementary.py`, `functional/__init__.py`
- `src/zepto/core/__init__.py`
- `tests/test_operation_contract.py`

### 6.1 `Substract` → `Subtract`

- Class `Substract` → `Subtract`; `family` `"substract"` → `"subtract"`.
- Functional wrapper `substract()` → `subtract()`.
- Update the four export sites (`operation/__init__.py`,
  `functional/elementary.py`, `functional/__init__.py`, `core/__init__.py`)
  and test references. No deprecation shim — the API is pre-release.

---

## Step 7 — Tests (plan §13, the subset unlocked by Steps 1–6)

### Files

- new `tests/test_metadata_accounting.py`
- extend `tests/test_operation_contract.py`
- extend `tests/test_structural_graph.py`

### `tests/test_metadata_accounting.py`

- `DType.itemsize` widths; `UNKNOWN` has `None` itemsize.
- `TensorMetadata` accepts/validates `dtype`, `role`, `persistent`,
  `requires_grad`; rejects non-`DType` dtype, non-`TensorRole` role,
  non-boolean persistent.
- `metadata_compatible`: unspecified expected dtype/role is a wildcard;
  specified must match exactly.
- `Parameter.trainable` defaults to `True` and is independent of
  `metadata.requires_grad`.
- `PrecisionPolicy.resolve` precedence: explicit dtype wins over
  semantic-type entry, which wins over role entry, which wins over default.
- `AccountingPolicy.bytes_for` computes `numel * itemsize`; raises on a
  policy that resolves to `UNKNOWN`.

### `tests/test_operation_contract.py` additions

- Validators report focused failures: bad family via
  `DeclarationValidator.validate_family`, unknown saved port via
  `InvocationValidator.validate_saved_state`, bad alias via
  `validate_aliases` (assert distinct, specific error messages).
- `Operation.validate_result()` does **not** invoke `resource_events()`
  (e.g. an operation whose `resource_events` raises still validates).
- `Subtract` family/backward contract.
- Gradient-accumulation event sequence without `ResourceEventKind.GRADIENT`.

### `tests/test_structural_graph.py` additions

- View shares storage; `CONTIGUOUS_COPY` allocates fresh storage (Step 3.4).
- `GraphValidator.validate_storage_links` rejects a hand-built graph whose
  copy output illegally shares input storage.
- `add_parameter(metadata, trainable=False)` round-trips through
  `graph.parameter()`.

Run: `pytest tests/ -q` after each step; the full suite must pass before
moving to the next step.

---

## Explicitly out of scope for this iteration

These follow the plan's dependency order and start only after Steps 1–7 land:

1. **Auxiliary ports (§3)** — `SemanticOperation.auxiliary_ports`,
   `OperationResult.auxiliary_outputs/auxiliary_aliases`,
   builder/composition changes. Depends on the validator split (Step 4) so
   port validation extends cleanly.
2. **Lowering (§8)** — `InvocationContext`, `LoweredTensor/Operation/Graph`,
   `lower()`. Depends on Steps 1–2 (dtype resolution) and Step 3 (correct
   storage identities).
3. **ReLU-like activation (§12, deferred)** — not a standalone structural
   `Operation`. ReLU is a variant of the existing `Maximum` operation: it
   stores the comparison mask until backward consumption, which differs from
   `Maximum`'s current saved-operand contract. Implement this as a *lowered*
   specialization of `Maximum` once lowering (§8) exists, under a name that
   expresses the mask-retention semantics rather than reintroducing a top-level
   `ReLU` class.
4. **Memory/FLOP accounting and `estimate()` (§9–10)** — depends on lowering.
5. **Optimizer policies (§11)** — depends on lowering and memory accounting.
6. **Real `LayerNorm`/`RMSNorm` compositions (§12)** — depends on auxiliary
   ports and the primitive set; both stay `Module` compositions built from
   `subtract`/`multiply`/`divide`/`sqrt` wrappers.
7. **Docs/ADR updates and full public-API exports (§14)** — after the API
   stabilizes.
