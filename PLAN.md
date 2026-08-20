# Issue #3: Migrate to concrete operation declarations

## Goal

Replace the current composed-contract implementation with a small, explicit
operation model:

```text
functional wrapper
    → immutable Operation instance
    → GraphCompositionContext.apply
    → StructuralGraphBuilder
    → StructuralOperation graph occurrence
```

Each primitive operation is a named class. The class is the source of truth for
both semantic behavior and estimation behavior. Functional wrappers remain the
ergonomic user API and do not contain operation definitions.

This plan changes implementation only. It does not implement backend execution,
lowering selection, fusion, or concrete device allocation.

## Target domain model

### Operation declaration

`Operation` is the public declaration base and directly implements both
behavioral interfaces:

```python
class Operation(SemanticOperation, EstimationOperation, ABC):
    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        return ()

    @abstractmethod
    def infer_result(
        self,
        inputs: tuple[TensorMetadata, ...],
        parameters: tuple[TensorMetadata, ...],
    ) -> OperationResult: ...

    def infer(
        self,
        inputs: tuple[TensorMetadata, ...],
        parameters: tuple[TensorMetadata, ...] = (),
    ) -> OperationResult:
        # shared tensor/parameter arity, inference, and result validation
        ...

    def validate_declaration(self) -> None:
        ...
```

`SemanticOperation` and `EstimationOperation` remain behavioral interfaces.
Concrete classes such as `Add`, `Multiply`, `MatMul`, `Reshape`,
`Transpose`, `ReLU`, and `Split` inherit `Operation`.

`StructuralOperation` remains a graph occurrence: it stores one immutable
`Operation`, bound tensor/parameter/output ports, graph tensor identities,
parameters, provenance, and the invocation `OperationResult`. It must not be
renamed or confused with the declaration class.

### Lowering relationship

The structural operation is backend-neutral. Lowering must support both:

```text
one structural operation  → many candidate implementations
many structural operations → one fused implementation
```

Candidate implementations and fused implementations are selected later by
lowering. They do not become subclasses of the structural `Operation` classes
and are outside this issue.

## Target package layout

Migrate the current flat modules into packages. Do not retain parallel
`operation.py`/`operation/` or `functional.py`/`functional/` APIs.

```text
src/zepto/
├── core/
│   ├── operation/
│   │   ├── __init__.py
│   │   ├── base.py          # SemanticOperation, EstimationOperation, Operation
│   │   ├── result.py        # OperationResult, AliasSpec, BackwardSpec
│   │   ├── estimation.py    # EstimationContext, ResourceEvent, event kinds
│   │   ├── validation.py    # reusable validation helpers
│   │   ├── add.py           # Add
│   │   ├── multiply.py      # Multiply
│   │   ├── matmul.py        # MatMul
│   │   ├── reshape.py       # Reshape
│   │   ├── transpose.py     # Transpose
│   │   ├── relu.py          # ReLU
│   │   └── split.py         # Split
│   │
│   └── functional/
│       ├── __init__.py
│       └── elementwise.py, matrix.py, view.py
│
└── modules/
    ├── __init__.py
    ├── layer_norm.py        # LayerNorm(Module)
    ├── rms_norm.py          # RMSNorm(Module)
    └── rope.py              # RoPE(Module)
```

`zepto.modules` depends on `zepto.core`; `zepto.core` must not import concrete
modules. The existing `Module` base may remain in `core.composition` during
this migration.

## Implementation phases

### 1. Establish the new operation package

- Create `core/operation/` and move the result, estimation, and interface
  records into focused modules.
- Implement `Operation` as the single complete declaration base.
- Make operation declarations immutable where they carry configuration:
  `Reshape(shape=...)`, `Transpose(permutation=...)`,
  and similar operations should use frozen dataclasses.
- Move `StructuralOperation` to the graph-facing location that avoids a
  declaration/node name collision.
- Export the new public types from `core/operation/__init__.py` and
  `core/__init__.py`.
- Remove `OperationContract`, `_FunctionalSemantic`, `_FunctionalEstimation`,
  `legacy_contract`, and `graph_contract`.
- Remove `OperationSpec` after all existing callers have been migrated.

### 2. Implement declaration and invocation validation

Keep validation backend-neutral and execute it before graph mutation.

#### Declaration validation

`Operation.validate_declaration()` checks only immutable declaration data:

```python
def validate_declaration(self) -> None:
    require_non_empty(self.family)
    inputs = require_tuple(self.input_ports, "input_ports")
    parameters = require_tuple(self.parameter_ports, "parameter_ports")
    outputs = require_tuple(self.output_ports, "output_ports")
    ports = (*inputs, *parameters, *outputs)
    require_instances(ports, PortSpec)
    require_unique_names(ports)
    require_value_kind(inputs, ValueKind.TENSOR)
    require_value_kind(parameters, ValueKind.PARAMETER)
    require_valid_value_kinds(outputs)
    require_valid_declared_metadata(ports)

    backward = self.backward
    require_instance(backward, BackwardSpec)
    known = {port.name for port in ports}
    require_subset(backward.saved_for_backward, known)
    require_subset(backward.gradient_inputs, known)
    require_subset(backward.gradient_outputs, known)
    if not backward.supported:
        require_empty_backward_requirements(backward)

    require_callable(self.infer_result)
    require_callable(self.forward_flops)
    require_callable(self.backward_flops)
    require_callable(self.resource_events)
```

This validation must not fabricate metadata or invoke estimation. It runs when
an operation is created or first applied, and again at the graph boundary.

#### Invocation/result validation

`infer()` is final/shared behavior. Concrete classes implement
`infer_result()` only:

```python
def infer(self, inputs, parameters=()):
    self.validate_declaration()
    require_input_metadata(inputs)
    require_arity(inputs, len(self.input_ports))
    require_parameter_metadata(parameters)
    require_arity(parameters, len(self.parameter_ports))
    result = self.infer_result(inputs, parameters)
    self.validate_result(inputs, parameters, result)
    return result
```

`validate_result()` checks:

```python
def validate_result(self, inputs, parameters, result):
    require_instance(result, OperationResult)
    require_arity(result.outputs, len(self.output_ports))
    require_output_metadata(result.outputs)
    require_alias_arity(result.aliases, len(result.outputs))

    input_names = {port.name for port in self.input_ports}
    for index, alias in enumerate(normalize_aliases(result)):
        if alias is None:
            continue
        require(alias.source_port in input_names)
        source = port_named(self.input_ports, alias.source_port)
        if alias.materialization is VIEW:
            require_view_compatible(source, result.outputs[index])
        elif alias.materialization is CONTIGUOUS_COPY:
            require_materializable(result.outputs[index])
        else:
            reject("unknown materialization mode")

    require_valid_saved_state(result.saved_for_backward, self.backward, self)

    context = EstimationContext(
        port_metadata=(
            *((port.name, metadata) for port, metadata in zip(
                self.input_ports, inputs, strict=True
            )),
            *((port.name, metadata) for port, metadata in zip(
                self.parameter_ports, parameters, strict=True
            )),
            *((port.name, metadata) for port, metadata in zip(
                self.output_ports, result.outputs, strict=True
            )),
        )
    )
    require_non_negative_integer(self.forward_flops(context, result))
    require_non_negative_integer(
        self.backward_flops(replace(context, phase="backward"), result)
    )
    events = self.resource_events(context, result)
    require_tuple(events, "resource_events")
    for event in events:
        require_valid_resource_event(event, self, result)
```

The exact error classes may remain focused in `core.errors`, but all failures
must occur before output tensors, storage, or consumer relationships are
allocated or mutated.

#### Graph-boundary validation

`StructuralGraphBuilder.add_operation()` must:

1. Require an `Operation`, not an opaque object.
2. Re-run `validate_declaration()`.
3. Validate operation family and bound port names/value kinds.
4. Validate input tensor and parameter ownership, arity, ordering, and metadata
   against separately declared tensor and parameter ports.
5. Validate concrete metadata and declared port contracts.
6. Infer only when no result was supplied; otherwise validate the supplied
   result against the builder's actual input metadata.
7. Re-run `validate_result()`.
8. Validate alias storage relationships and materialization.
9. Allocate graph identities only after every check succeeds.

`StructuralGraph.validate()` repeats declaration/result, graph-reference,
arity, ordering, ownership, and storage invariants for finalized graphs.

### 3. Add concrete primitive operation classes

Implement each operation in its own file. Each class owns ports, inference,
estimation, resource events, and backward metadata.

- `Identity`: same metadata, view/alias result, zero FLOPs.
- `Add`: elementwise broadcast inference and one output allocation.
- `Multiply`: same broadcast rules as `Add`, with its own product-rule
  backward declaration and elementwise FLOPs.
- `MatMul`: rank/contracting-dimension checks and
  `2 * M * K * N` theoretical FLOPs.
- `Reshape`: view-compatible output metadata and zero FLOPs.
- `Transpose`: permutation validation, view semantics, and zero FLOPs.
- `ReLU`: elementwise semantic behavior that saves the post-activation output
  used to recover the backward mask.
- `Split`: ordered multi-output inference and explicit output ordering.

Preserve semantic types and concrete dimensions while inferring. Do not encode
backend or device behavior in these classes.

#### 3.1 Correct the Atto comparison findings

Make parameter participation part of the operation contract instead of leaving
`StructuralOperation.parameter_ids` as an untyped side channel:

- Add `Operation.parameter_ports`, defaulting to `()`. Parameter ports are
  ordered `PortSpec` values with `value_kind=ValueKind.PARAMETER`.
- Extend `Operation.infer()`/`validate_result()` to receive parameter metadata
  separately from tensor input metadata. Build one `EstimationContext` whose
  named metadata contains tensor inputs, parameters, and outputs.
- Extend `StructuralOperation` and `StructuralGraphBuilder.add_operation()` with
  bound `parameter_ports`; require one owned `ParameterId` per declared port and
  reject missing, extra, reordered, or shape-incompatible parameters before any
  graph mutation.
- Use `Parameter.requires_grad` as the canonical parameter gradient flag. When
  binding a parameter port into inference or estimation, copy that value into
  the bound `TensorMetadata.require_grad`; do not rely on the metadata originally
  passed to `add_parameter()`.
- Validate that backward references may target tensor inputs, parameter inputs,
  or outputs. Resolve saved parameter references to parameter ports rather than
  pretending that parameters are tensors.
- Extend `PortRef.role` with `"parameter"` and make `PortRef.resolve()` select
  `StructuralOperation.parameter_ports` for that role.

Implement the four operation fixes as follows:

1. `src/zepto/core/operation/multiply.py`
   - Stop inheriting the complete `Add` contract. Reuse only a private
     `_broadcast_shape()` helper shared with `add.py`.
   - Declare both `left` and `right` in `saved_for_backward`, because each
     product-rule gradient needs the opposite operand.
   - Return `saved_for_backward=("left", "right")` from `infer_result()`.
   - Set output `require_grad` to
     `left.require_grad or right.require_grad`.
   - Charge one output-sized multiplication for each operand requiring a
     gradient. If broadcast-gradient reductions are modeled in this issue,
     additionally charge `output_numel - operand_numel` additions for each
     reduced operand; otherwise document that reductions remain a lowering
     estimate and test the multiplication count explicitly.

2. `src/zepto/core/operation/matmul.py`
   - Replace the unconditional output `require_grad=True` with
     `left.require_grad or right.require_grad`.
   - Keep both operands saved for backward.
   - Keep one forward-sized GEMM per operand whose bound metadata requires a
     gradient.

3. `src/zepto/core/operation/relu.py`
   - Change `BackwardSpec.saved_for_backward` and
     `OperationResult.saved_for_backward` from `("input",)` to `("output",)`.
   - Preserve output `require_grad` from the input.
   - Keep forward and backward at one FLOP per output element. This matches
     Atto's `output > 0` mask and its activation lifetime behavior.

Do not add decode-specific formulas copied from Atto. Zepto represents a
different concrete shape scenario by rebuilding the structural graph, as
recorded in ADR 0005.

### 4. Replace functional implementations with thin wrappers

Move wrappers into `core/functional/`. Each wrapper should only:

1. Require an active `GraphCompositionContext`.
2. Construct a named operation instance.
3. Forward graph tensors and parameters to `context.apply()`.

Examples:

```python
def add(left, right):
    return _require_context().apply(Add(), left, right)


```

Add `multiply` as a first-class wrapper and operation. Preserve public import
compatibility where practical, but make the named classes the canonical API.

### 5. Update composition and graph integration

- Change `GraphCompositionContext.apply()` to accept `Operation`.
- Call `operation.infer()` exactly once during eager composition.
- Resolve parameter IDs to metadata, bind tensor/parameter/output metadata to
  copied `PortSpec` instances, and pass both metadata groups to `infer()`.
- Pass the same operation and result to `add_operation()`.
- Store the operation instance in each `StructuralOperation`.
- Store parameter ports alongside parameter IDs so graph validation can prove
  their names, order, value kinds, metadata, and gradient participation.
- Remove compatibility paths that silently turn incomplete declarations into
  zero-cost legacy operations.
- Ensure multi-output return values preserve declared output order.
- Preserve module provenance independently from operation class identity.

### 6. Move reusable modules outside core

Create `src/zepto/modules/` with one file per reusable composition:

- `LayerNorm`: compose primitive reduction, arithmetic, and activation/view
  operations as supported by the available primitive set.
- `RMSNorm`: compose primitive multiply, reduction, add, reciprocal-square-root,
  and multiply operations.
- `RoPE`: compose primitive shape, arithmetic, trigonometric, and view
  operations as the primitive library permits.

These classes are `Module` compositions, not primitive `Operation` classes and
not structural graph nodes. Their provenance should identify the module path so
future many-to-one lowering can match the resulting region.

### 7. Revise documentation and domain records

- Update `docs/adr/0004-complete-operation-contracts.md` from composed
  `OperationContract` to the direct `Operation` hierarchy.
- Keep `CONTEXT.md` terminology aligned:
  `Operation` is the semantic declaration, while `StructuralOperation` is its
  graph occurrence.
- Document that implementations are independently selected during lowering,
  including one-to-many candidate selection and many-to-one fusion.
- Remove stale references to `SemanticContract`, `EstimationContract`, and
  opaque operation declarations.

## Tests

Update existing graph tests and add focused tests covering:

- Every concrete primitive is an `Operation` and implements both interfaces.
- Incomplete subclasses cannot be applied to a graph.
- Invalid families, ports, kinds, backward references, and metadata fail at
  declaration validation.
- Inference is called once and returns a fully validated `OperationResult`.
- Invalid arity, output count, metadata, aliases, views, and saved state fail
  before graph mutation.
- Negative/non-integer FLOPs and invalid resource events are rejected.
- `Add` and `Multiply` broadcast correctly and reject incompatible shapes.
- `Multiply` saves both operands, propagates `require_grad`, and charges only
  the requested product-rule gradients.
- `MatMul` uses `2 * M * K * N`, propagates `require_grad` from its operands,
  and returns a non-gradient output for two non-gradient operands.
- `ReLU` saves its output rather than its input and resolves that saved value to
  an output `PortRef`.
- `Reshape`, `Transpose`, and `Identity` preserve storage when they declare a
  view/alias.
- `Split` preserves multi-output order.
- Graph nodes retain the concrete operation instance and invocation result.
- Builder-level validation catches a deliberately altered declaration.
- Finalized graph validation catches invalid ownership/order/storage state.
- Functional wrappers remain ergonomic and produce the expected operation
  classes.
- `LayerNorm`, `RMSNorm`, and `RoPE` live outside `core` and create primitive
  operation regions rather than composite operation nodes.

## Verification commands

Run from the repository root:

```bash
pytest -q
pytest -q tests/test_operation_contract.py
python -m compileall src/zepto
```

Also verify that imports resolve only through the new package layout and that
no stale `OperationContract`, `_FunctionalSemantic`, `_FunctionalEstimation`,
`legacy_contract`, or `graph_contract` implementation remains.

## Completion checklist

- [ ] New operation package exists and exports the direct `Operation` hierarchy.
- [ ] Old composed-contract and legacy-adapter paths are removed.
- [ ] Declaration and result validation are implemented at all three stated
  boundaries.
- [ ] Named primitive classes replace anonymous functional contracts.
- [ ] Functional wrappers are thin and include `multiply`.
- [ ] Parameter ports are explicit, bound, and validated independently from
  tensor input ports.
- [ ] Multiply, MatMul, and ReLU match the corrected forward/backward
  contracts described in section 3.1.
- [ ] Graph nodes store concrete operation declarations and results.
- [ ] Reusable modules are in sibling `zepto/modules/` files.
- [ ] ADR and stale terminology are updated.
- [ ] Focused and full test suites pass.
