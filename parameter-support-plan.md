# Parameter Support Plan

Concrete implementation plan for making Zepto **parameter-aware** and for
clarifying the type model around weights vs data-flow values. This document
combines:

1. **Vocabulary and composition API** — `ValueMetadata`, `GraphParameter`, two-plane model
2. **Operation contract** — `parameter_ports`, inference, validation, backward resolution
3. **Graph integration** — binding, storage, lowering hooks
4. **First consumer** — `LinearMatMul` operation + `Linear` module

Operations, structural graphs, and per-op lowering are already in place. This
plan does **not** implement estimation reports, fused-kernel lowering, or new
primitives beyond what is needed to prove the parameter path (`Linear`).

---

## Goals

| Goal | Success criterion |
|------|-------------------|
| Parameters participate in the operation contract | Ops declare `parameter_ports`; inference receives parameter metadata |
| Composition API is symmetric | `GraphTensor` for flow, `GraphParameter` for model state |
| Vocabulary is drawable on one diagram | Two-plane model documented; `ValueMetadata` shared description |
| Weight-bearing modules compose | `Linear` module builds a valid structural graph with bound parameter ports |
| Lowering can see parameters | `build_estimation_context()` includes parameter port metadata |
| LoRA / frozen-base ready | `trainable` on `Parameter`; `requires_grad` derived at bind time |

Non-goals for this plan:

- Renaming `Tensor` → `FlowTensor` (clarity gain too small vs breakage)
- Full `LayerNorm` / `RMSNorm` / `RoPE` math (Phase B — needs reduction primitives)
- `estimate()` / memory reports (Phase D)
- Module-scoped implementation pins / fusion (Phase C)

---

## Architectural model

Zepto stores **two parallel planes** in every structural graph:

```text
Description layer          ValueMetadata (shape, dtype, semantic_type, …)
                                   │
Composition handles        GraphTensor              GraphParameter
                                   │                        │
Graph storage              Tensor (flow)            Parameter (asset)
                                   │                        │
                         producer/consumers          referenced by ops
                         graph.tensors               graph.parameters
```

### Type responsibilities

| Type | Plane | Role |
|------|-------|------|
| **`ValueMetadata`** | Description | Shared shape/dtype/accounting for any array-like value |
| **`GraphTensor`** | Composition | Handle for per-invocation data-flow values in `forward()` |
| **`GraphParameter`** | Composition | Handle for model-persistent weights in `__init__()` |
| **`Tensor`** | Storage (flow) | DAG node with producer/consumers in `graph.tensors` |
| **`Parameter`** | Storage (asset) | Static weight in `graph.parameters`; referenced, not produced |

### Why keep a separate parameter registry

Metadata-only weights (`ValueMetadata.role = PARAMETER` on graph inputs) can
express dtype and accounting hints, but they cannot structurally separate:

- **Invocation inputs** (`build_graph(module, batch_metadata)`) from **model weights**
- **Optimizer state attachment** (future) from activation lifetimes
- **Parameter port contracts** (`ValueKind.PARAMETER`) from generic tensor inputs

The registry enforces these boundaries at the schema level. `ValueMetadata` is
shared; **container type** (`Tensor` vs `Parameter`) encodes graph role.

### `trainable` vs `requires_grad`

Per ADR 0006:

| Flag | Lives on | Meaning |
|------|----------|---------|
| **`trainable`** | `GraphParameter` / `Parameter` | Optimizer updates this weight (LoRA adapter yes, frozen base no) |
| **`requires_grad`** | `ValueMetadata` on flow values | Backward flows through this activation |

At parameter-port bind time, derive bound metadata:

```python
bound = replace(
    param.metadata,
    requires_grad=param.trainable,
    role=TensorRole.PARAMETER,
    persistent=True,
)
```

---

## Phase 0 — Vocabulary (`ValueMetadata`)

**Files:** `src/zepto/core/metadata.py`, `src/zepto/core/__init__.py`, `CONTEXT.md`

**Status:** `ValueMetadata` rename landed — `TensorMetadata` removed across the codebase.

### 0.1 Rename `TensorMetadata` → `ValueMetadata`

Replace the old type name everywhere. There is no backward-compatible alias;
`ValueMetadata` is the sole public name.

```python
# src/zepto/core/metadata.py

@dataclass(frozen=True, slots=True)
class ValueMetadata:
    """Backend-neutral shape and semantic metadata for a graph value."""

    shape: Shape
    semantic_type: str = "tensor"
    requires_grad: bool = False
    dtype: DType | None = None
    role: TensorRole | None = None
    persistent: bool = False

    # ... __post_init__ and helpers unchanged ...


def metadata_compatible(
    expected: ValueMetadata,
    actual: ValueMetadata,
) -> bool:
    ...
```

**Reason:** `Parameter` embeds the same metadata type as `Tensor`. The old name
`TensorMetadata` implied "parameter is a tensor node." `ValueMetadata` names
the shared description layer honestly.

### 0.2 Export `ValueMetadata` from `core`

```python
# src/zepto/core/__init__.py
from .metadata import ValueMetadata, TensorRole, metadata_compatible

__all__ = [..., "ValueMetadata", ...]
```

### 0.3 Add glossary entries to `CONTEXT.md`

Add definitions for **ValueMetadata**, **GraphTensor**, **GraphParameter**,
**Parameter**, and the two-plane diagram above. Update the **Operation**
definition to mention parameter ports:

> A reusable semantic operation declaration that maps input tensors and
> parameters to output tensors according to a mathematical operation and
> complete semantic and estimation behavior.

---

## Phase 1 — `GraphParameter` composition handle

**Files:** `src/zepto/core/composition.py`, `src/zepto/core/__init__.py`

### 1.1 New type

```python
# src/zepto/core/composition.py

@dataclass(frozen=True, slots=True)
class GraphParameter:
    """Symbolic parameter handle registered during module construction."""

    id: ParameterId
    metadata: ValueMetadata
    trainable: bool = True
```

**Reason:** Mirrors `GraphTensor`. Users should never hold bare `ParameterId`
in module code — the handle carries metadata and trainability for IDE clarity
and functional wrappers.

### 1.2 Fix and extend `GraphCompositionContext.parameter()`

Current code is broken (`requires_grad=` passed to `add_parameter(trainable=)`):

```python
# BEFORE (broken)
def parameter(self, metadata: ValueMetadata, *, requires_grad: bool = True) -> ParameterId:
    return self.builder.add_parameter(metadata, requires_grad=requires_grad)

# AFTER
def parameter(
    self,
    metadata: ValueMetadata,
    *,
    trainable: bool = True,
) -> GraphParameter:
    """Declare a model parameter in the owned graph."""
    parameter_id = self.builder.add_parameter(metadata, trainable=trainable)
    return GraphParameter(parameter_id, metadata, trainable=trainable)
```

### 1.3 Update `Module` to store `GraphParameter`

```python
class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, GraphParameter] = {}
        self._modules: dict[str, Module] = {}

    def register_parameter(self, name: str, parameter: GraphParameter) -> None:
        ...

    def named_parameters(
        self, prefix: str = ""
    ) -> Iterable[tuple[str, GraphParameter]]:
        ...
```

**Reason:** `named_parameters()` should return handles, not opaque IDs. Matches
PyTorch's `nn.Parameter` ergonomics.

---

## Phase 2 — Operation contract: `parameter_ports`

**Files:** `src/zepto/core/operation/base.py`, `validation.py`, `structural.py`,
`ports.py`, `graph.py`

### 2.1 Extend `Operation` base

Add a default-empty `parameter_ports` property and extend inference:

```python
# src/zepto/core/operation/base.py

class SemanticOperation(ABC):
    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        """Return ordered parameter port declarations."""
        return ()

    # infer_outputs stays inputs-only for ops without parameters
    ...


class Operation(SemanticOperation, EstimationOperation, ABC):
    def infer_result(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> OperationResult:
        self.validate_declaration()
        _INVOCATION_VALIDATOR.validate_inputs(self, inputs)
        _INVOCATION_VALIDATOR.validate_parameters(self, parameters)
        outputs = self.infer_outputs(inputs, parameters)
        ...
        self.validate_result(inputs, parameters, result)
        return result

    def validate_result(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...],
        result: OperationResult,
    ) -> None:
        _INVOCATION_VALIDATOR.validate_result(self, inputs, parameters, result)
```

For operations without parameters, provide a backward-compatible default:

```python
def infer_outputs(
    self,
    inputs: tuple[ValueMetadata, ...],
    parameters: tuple[ValueMetadata, ...] = (),
) -> tuple[ValueMetadata, ...]:
    # Existing ops override with inputs-only signature initially;
    # migrate each to accept parameters=() explicitly.
    raise NotImplementedError
```

**Migration strategy:** Add `parameters=()` to each concrete `infer_outputs`,
`saved_for_backward`, etc. Existing bodies ignore the argument. No behavior
change for primitive ops until `LinearMatMul` lands.

### 2.2 Extend declaration validation

```python
# src/zepto/core/operation/validation.py — DeclarationValidator.validate_ports

def validate_ports(self, operation: Operation) -> None:
    inputs = operation.input_ports
    parameters = operation.parameter_ports
    outputs = operation.output_ports
    auxiliary = operation.auxiliary_ports()
    ports = (*inputs, *parameters, *outputs, *auxiliary)
    ...
    for port in parameters:
        if port.value_kind is not ValueKind.PARAMETER:
            raise OperationError(
                f"Parameter port {port.name!r} must use ValueKind.PARAMETER"
            )
```

Extend `validate_backward` known-port set to include `parameter_ports`.

Extend `validate_saved_state` to allow saved names from parameter ports:

```python
known = {
    port.name
    for port in (
        *operation.input_ports,
        *operation.parameter_ports,
        *operation.output_ports,
    )
}
```

### 2.3 Add parameter invocation validation

```python
class InvocationValidator:
    def validate_parameters(
        self,
        operation: Operation,
        parameters: tuple[ValueMetadata, ...],
    ) -> None:
        if len(parameters) != len(operation.parameter_ports):
            raise OperationError(
                f"{operation.family!r} expects "
                f"{len(operation.parameter_ports)} parameters, got {len(parameters)}"
            )
        if not all(isinstance(value, ValueMetadata) for value in parameters):
            raise OperationError("Operation parameters must be ValueMetadata")
```

### 2.4 Extend `PortRef.role`

```python
# src/zepto/core/ports.py

@dataclass(frozen=True, slots=True)
class PortRef:
    operation_id: OperationId
    port_name: str
    role: Literal["input", "output", "auxiliary", "parameter"]

    def resolve(self, operation: StructuralOperation) -> PortSpec:
        ...
        if self.role == "parameter":
            ports = operation.parameter_ports
        elif self.role == "input":
            ports = operation.input_ports
        ...
```

### 2.5 Extend `StructuralOperation`

```python
@dataclass(frozen=True, slots=True)
class StructuralOperation:
    input_ports: tuple[PortSpec, ...]
    parameter_ports: tuple[PortSpec, ...]          # NEW — bound metadata
    output_ports: tuple[PortSpec, ...]
    input_tensors: tuple[TensorId, ...]
    parameter_ids: tuple[ParameterId, ...]           # validated against parameter_ports
    output_tensors: tuple[TensorId, ...]
    ...
```

### 2.6 Parameter metadata resolution helper

```python
# src/zepto/core/composition.py (or graph.py)

def bound_parameter_metadata(
    parameter: Parameter,
) -> ValueMetadata:
    """Return metadata for inference/estimation at a parameter port."""
    return replace(
        parameter.metadata,
        requires_grad=parameter.trainable,
        role=TensorRole.PARAMETER,
        persistent=True,
    )
```

**Reason:** Single place for ADR 0006 trainability → requires_grad copy rule.
PLAN.md §3.1 requires this explicitly.

---

## Phase 3 — Composition and graph binding

**Files:** `composition.py`, `graph.py`, `functional/_common.py`

### 3.1 Extend `GraphCompositionContext.apply()`

```python
def apply(
    self,
    operation: Operation,
    *inputs: GraphTensor,
    parameters: tuple[GraphParameter, ...] = (),
    module_path: tuple[str, ...] | None = None,
    component_type: str | None = None,
    source_label: str | None = None,
) -> GraphTensor | tuple[GraphTensor, ...]:
    input_metadata = tuple(value.metadata for value in inputs)

    parameter_metadata = tuple(
        bound_parameter_metadata(self.builder._parameters[param.id])
        for param in parameters
    )

    result = operation.infer_result(input_metadata, parameter_metadata)
    output_metadata = result.outputs

    input_ports, parameter_ports, output_ports = self._bind_ports(
        operation,
        input_metadata,
        parameter_metadata,
        output_metadata,
    )
    ...
    output_ids = self.builder.add_operation(
        operation_family=operation.family,
        input_ports=input_ports,
        parameter_ports=parameter_ports,
        output_ports=output_ports,
        input_tensors=tuple(value.id for value in inputs),
        parameter_ids=tuple(param.id for param in parameters),
        output_metadata=output_metadata,
        provenance=provenance,
        operation=operation,
        result=result,
    )
    ...
```

### 3.2 Extend `_bind_ports()`

```python
def _bind_ports(
    self,
    operation: Operation,
    input_metadata: tuple[ValueMetadata, ...],
    parameter_metadata: tuple[ValueMetadata, ...],
    output_metadata: tuple[ValueMetadata, ...],
) -> tuple[tuple[PortSpec, ...], tuple[PortSpec, ...], tuple[PortSpec, ...]]:
    if len(operation.parameter_ports) != len(parameter_metadata):
        raise PortArityError(...)

    bound_parameters = tuple(
        self._bind_port(operation, port, metadata, direction="parameter")
        for port, metadata in zip(
            operation.parameter_ports, parameter_metadata, strict=True
        )
    )
    ...
    return bound_inputs, bound_parameters, bound_outputs
```

### 3.3 Extend `StructuralGraphBuilder.add_operation()`

```python
def add_operation(
    self,
    *,
    operation_family: str,
    input_ports: tuple[PortSpec, ...],
    parameter_ports: tuple[PortSpec, ...] = (),      # NEW
    output_ports: tuple[PortSpec, ...],
    input_tensors: tuple[TensorId, ...],
    parameter_ids: tuple[ParameterId, ...] = (),
    output_metadata: tuple[ValueMetadata, ...],
    provenance: Provenance,
    operation: Operation,
    result: OperationResult | None = None,
) -> tuple[TensorId, ...]:
    if len(parameter_ports) != len(parameter_ids):
        raise PortArityError("Parameter port and id counts differ")

    # Validate bound ports match declaration (including parameter_ports)
    if (
        tuple((p.name, p.value_kind) for p in operation.parameter_ports)
        != tuple((p.name, p.value_kind) for p in parameter_ports)
    ):
        raise OperationError("Bound parameter ports do not match declaration")

    parameter_metadata = tuple(
        bound_parameter_metadata(self._parameters[pid])
        for pid in parameter_ids
    )

    if result is None:
        result = operation.infer_result(input_metadata, parameter_metadata)
    operation.validate_result(input_metadata, parameter_metadata, result)

    parameter_port_names = {port.name for port in parameter_ports}
    saved_for_backward = tuple(
        PortRef(
            operation_id,
            saved_name,
            (
                "input" if saved_name in input_port_names
                else "parameter" if saved_name in parameter_port_names
                else "output"
            ),
        )
        for saved_name in result.saved_for_backward
    )

    self._operations[operation_id] = StructuralOperation(
        ...
        parameter_ports=parameter_ports,
        parameter_ids=parameter_ids,
        ...
    )
```

**Reason:** Three-boundary validation (declaration, invocation, graph build)
already exists for tensor ports. Parameter ports get the same treatment.

---

## Phase 4 — First parameter-bearing operation: `LinearMatMul`

**Files:** `src/zepto/core/operation/linear_matmul.py`, `functional/matrix.py`

We introduce a **new operation family** rather than overloading `MatMul`, because
`MatMul` today declares two tensor input ports. Mixing parameter and activation
ports under one family would break existing port contracts and tests.

### 4.1 Operation declaration

```python
# src/zepto/core/operation/linear_matmul.py

"""Linear layer matrix multiply: activation × weight parameter."""

from ..metadata import ValueMetadata, TensorRole
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import numel
from .matmul import _broadcast_batch  # reuse batch logic
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class LinearMatMul(Operation):
    """Multiply an activation by a weight parameter along the last two dims."""

    @property
    def family(self) -> str:
        return "linear_matmul"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(
                "weight",
                value_kind=ValueKind.PARAMETER,
                metadata=ValueMetadata((), semantic_type="weight"),
            ),
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input", "weight"),
            gradient_inputs=("output",),
            gradient_outputs=("input", "weight"),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (activation,) = inputs
        (weight,) = parameters
        if len(activation.shape) < 2:
            raise OperationError("linear_matmul requires rank >= 2 activation")
        if len(weight.shape) != 2:
            raise OperationError("linear_matmul weight must be rank 2")
        if activation.shape[-1] != weight.shape[0]:
            raise OperationError(
                f"in_features mismatch: activation {activation.shape[-1]} "
                f"vs weight {weight.shape[0]}"
            )
        batch = _broadcast_batch(activation, weight, family=self.family)
        out_shape = (*batch, activation.shape[-2], weight.shape[1])
        requires_grad = activation.requires_grad or weight.requires_grad
        return (
            ValueMetadata(
                out_shape,
                requires_grad=requires_grad,
                role=TensorRole.ACTIVATION,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        outputs: tuple[ValueMetadata, ...],
    ) -> tuple[str, ...]:
        activation, = inputs
        (weight,) = inputs  # NOTE: use parameters in full impl
        saved: list[str] = []
        if activation.requires_grad:
            saved.append("weight")
        # weight gradient needs activation; activation gradient needs weight
        (output,) = outputs
        if output.requires_grad:
            saved.extend(["input", "weight"])
        return tuple(dict.fromkeys(saved))  # preserve order, dedupe

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.metadata_for("output")
        return 2 * numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        # One GEMM per operand requiring a gradient (same convention as MatMul)
        ...

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        ...
```

> **Note:** The `saved_for_backward` sketch above must receive `parameters`
> separately once the hook signature is extended. Shown here to illustrate
> intent: weight is saved via a **parameter port**, not an input tensor.

### 4.2 Functional wrapper

```python
# src/zepto/core/functional/matrix.py

from ..composition import GraphParameter, GraphTensor
from ..operation import LinearMatMul
from ._common import context


def linear_matmul(
    input: GraphTensor,
    weight: GraphParameter,
) -> GraphTensor:
    """Record activation × weight parameter multiplication."""
    return context().apply(LinearMatMul(), input, parameters=(weight,))
```

**Reason:** Functional wrappers stay thin — construct op, forward to `apply()`.
Parameter handles use the `parameters=` kwarg, keeping positional args for
activations only.

---

## Phase 5 — First parameter-bearing module: `Linear`

**Files:** `src/zepto/modules/linear.py`, `src/zepto/modules/__init__.py`

```python
# src/zepto/modules/linear.py

"""Linear layer module."""

from ..core.composition import (
    GraphCompositionContext,
    GraphParameter,
    GraphTensor,
    Module,
)
from ..core.functional import add, linear_matmul
from ..core.metadata import ValueMetadata


class Linear(Module):
    """Affine transform: linear_matmul(x, weight) + optional bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        ctx = GraphCompositionContext.current()
        if ctx is None:
            raise RuntimeError("Linear requires an active GraphCompositionContext")

        weight = ctx.parameter(
            ValueMetadata(
                (in_features, out_features),
                semantic_type="weight",
            )
        )
        self.register_parameter("weight", weight)

        self.bias: GraphParameter | None = None
        if bias:
            bias_param = ctx.parameter(
                ValueMetadata((out_features,), semantic_type="bias")
            )
            self.register_parameter("bias", bias_param)
            self.bias = bias_param

    def forward(self, x: GraphTensor) -> GraphTensor:
        out = linear_matmul(x, self._parameters["weight"])
        if self.bias is not None:
            # bias add uses existing Add op — bias passed as activation input
            # until a dedicated bias-add-with-parameter op exists, OR:
            # extend Add with optional parameter port for bias (future)
            out = add(out, self._wrap_bias_as_graph_tensor())
        return out
```

**Open choice — bias handling:**

| Option | Pros | Cons |
|--------|------|------|
| A. `Add(out, bias_as_GraphTensor)` via helper that re-exposes parameter as graph constant | Minimal new ops | Bias not on parameter port for add op |
| B. New `LinearAddBias` op with bias parameter port | Full contract | More ops |
| C. Defer bias; ship weight-only `Linear` first | Smallest diff | Incomplete layer |

**Recommendation:** Ship **Option C** first (weight-only `Linear`), then add
`BiasAdd` with a parameter port in a follow-up PR. Keeps this plan focused on
proving the parameter path.

### End-to-end composition example

```python
from zepto.core import ValueMetadata, build_graph
from zepto.modules import Linear

class MLP(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = Linear(dim, dim)

    def forward(self, x: GraphTensor) -> GraphTensor:
        return self.fc(x)

# Module construction must happen inside the same context as build_graph,
# OR Linear accepts pre-built parameters — see §5.1 below.

graph = build_graph(
    MLP(dim=512),
    (ValueMetadata((32, 128, 512)),),  # batch invocation input only
)
assert len(graph.parameters) == 1
op = graph.operation(graph.operations[0])
assert op.parameter_ports[0].name == "weight"
assert op.parameter_ids[0] in graph.parameters
```

### 5.1 Module construction context

**Problem:** `Linear.__init__` calls `GraphCompositionContext.current()`, but
`build_graph()` opens the context *after* module instantiation:

```python
# Current build_graph — module created OUTSIDE context
def build_graph(module, input_metadata):
    with GraphCompositionContext() as context:
        inputs = tuple(context.input(m) for m in input_metadata)
        outputs = module(*inputs)   # parameters must already exist
```

**Fix — lazy parameter registration:**

```python
def build_graph(
    module_factory: Callable[[GraphCompositionContext], Module],
    input_metadata: tuple[ValueMetadata, ...],
) -> StructuralGraph:
    with GraphCompositionContext() as context:
        module = module_factory(context)
        inputs = tuple(context.input(m) for m in input_metadata)
        outputs = module(*inputs)
        ...
```

Usage:

```python
graph = build_graph(
    lambda ctx: MLP(dim=512),
    (ValueMetadata((32, 128, 512)),),
)
```

**Alternative:** Keep `build_graph(module, ...)` and require modules to declare
parameters in a `setup(context)` hook called by `build_graph` before `forward`.

**Recommendation:** Factory form is simplest and matches "construct inside
context." Keep a backward-compatible overload that accepts a pre-built module
with no parameters for existing tests.

---

## Phase 6 — Lowering hooks

**Files:** `src/zepto/core/lowering/helpers.py`,
`src/zepto/core/lowering/implementations/identity.py`

### 6.1 Include parameters in `build_estimation_context()`

```python
def build_estimation_context(
    structural: StructuralOperation,
    graph: StructuralGraph,
    context: InvocationContext,
) -> EstimationContext:
    port_metadata: list[tuple[str, ValueMetadata]] = []

    for port, tensor_id in zip(
        structural.input_ports, structural.input_tensors, strict=True
    ):
        metadata = resolve_tensor_metadata(graph.tensor(tensor_id).metadata, context)
        port_metadata.append((port.name, metadata))

    for port, parameter_id in zip(
        structural.parameter_ports, structural.parameter_ids, strict=True
    ):
        param = graph.parameter(parameter_id)
        metadata = resolve_tensor_metadata(
            bound_parameter_metadata(param), context
        )
        port_metadata.append((port.name, metadata))

    assert structural.result is not None
    for port, metadata in zip(
        structural.output_ports, structural.result.outputs, strict=True
    ):
        port_metadata.append((port.name, resolve_tensor_metadata(metadata, context)))

    return EstimationContext(
        phase=context.phase,
        port_metadata=tuple(port_metadata),
        precision=context.precision,
        state=context.state,
    )
```

### 6.2 Register `linear_matmul` in default registry

```python
# src/zepto/core/lowering/defaults.py
DEFAULT_REGISTRY = LoweringRegistry([
    ...
    ImplementationDescriptor(
        family="linear_matmul",
        implementation=IdentityImplementation(),  # 1:1 passthrough initially
        priority=0,
    ),
])
```

**Reason:** Unblocks `lower()` on parameter-bearing graphs without waiting for
fused linear kernels.

---

## Phase 7 — Tests

**New file:** `tests/test_parameters.py`

### 7.1 Unit tests

```python
def test_graph_parameter_round_trips_through_context() -> None:
    with GraphCompositionContext() as ctx:
        param = ctx.parameter(ValueMetadata((4, 8)), trainable=False)
        assert isinstance(param, GraphParameter)
        assert param.trainable is False

def test_apply_resolves_parameter_metadata() -> None:
    ...

def test_linear_matmul_validates_in_features() -> None:
    ...

def test_add_operation_rejects_parameter_port_mismatch() -> None:
    ...

def test_saved_for_backward_resolves_parameter_role() -> None:
    ...
```

### 7.2 Integration test

```python
def test_linear_module_builds_parameter_bound_graph() -> None:
    def make_model(ctx: GraphCompositionContext) -> Linear:
        return Linear(512, 256)

    graph = build_graph(
        make_model,
        (ValueMetadata((2, 128, 512)),),
    )
    assert len(graph.parameters) == 1
    weight = graph.parameter(next(iter(graph.parameters)))
    assert weight.metadata.shape == (512, 256)

    op = graph.operation(next(iter(graph.operations)))
    assert op.operation_family == "linear_matmul"
    assert len(op.parameter_ports) == 1
    assert op.parameter_ports[0].value_kind is ValueKind.PARAMETER
```

### 7.3 Regression

- All existing tests pass unchanged (primitives have `parameter_ports=()`)
- `test_builder_preserves_order_and_shared_parameters` updated to use bound
  `parameter_ports` once a test operation declares them

---

## Implementation order

```text
Phase 0  ValueMetadata rename + CONTEXT.md glossary
   ↓
Phase 1  GraphParameter + fix context.parameter()
   ↓
Phase 2  Operation.parameter_ports + validation + PortRef
   ↓
Phase 3  apply() + add_operation() binding
   ↓
Phase 4  LinearMatMul operation + functional wrapper
   ↓
Phase 5  Linear module + build_graph factory fix
   ↓
Phase 6  build_estimation_context() + registry entry
   ↓
Phase 7  tests/test_parameters.py
```

Each phase is independently mergeable. Phases 0–3 are the **parameter foundation**.
Phases 4–7 prove the path with one real module.

---

## File change summary

| File | Change |
|------|--------|
| `src/zepto/core/metadata.py` | Rename `TensorMetadata` → `ValueMetadata` |
| `src/zepto/core/composition.py` | `GraphParameter`, fix `parameter()`, extend `apply()`, `build_graph` factory |
| `src/zepto/core/parameter.py` | Use `ValueMetadata` type hint |
| `src/zepto/core/operation/base.py` | `parameter_ports`, extend `infer_result` |
| `src/zepto/core/operation/validation.py` | Parameter validation, backward port set |
| `src/zepto/core/operation/structural.py` | Add `parameter_ports` field |
| `src/zepto/core/ports.py` | `PortRef.role` += `"parameter"` |
| `src/zepto/core/graph.py` | `add_operation()` parameter binding, saved_for_backward resolution |
| `src/zepto/core/operation/linear_matmul.py` | **New** — first parameter op |
| `src/zepto/core/functional/matrix.py` | `linear_matmul()` wrapper |
| `src/zepto/modules/linear.py` | **New** — first weight-bearing module |
| `src/zepto/core/lowering/helpers.py` | Parameter ports in estimation context |
| `src/zepto/core/lowering/defaults.py` | Register `linear_matmul` |
| `src/zepto/core/__init__.py` | Export new public types |
| `CONTEXT.md` | Glossary + two-plane model |
| `tests/test_parameters.py` | **New** — parameter contract tests |

---

## Design decisions recap

| Decision | Reason |
|----------|--------|
| `ValueMetadata` as sole metadata type | Clarifies shared description; direct rename, no alias |
| `GraphParameter` handle | Symmetric with `GraphTensor`; no bare `ParameterId` in user code |
| Separate `LinearMatMul` family | Avoids breaking `MatMul`'s two-tensor-input contract |
| Keep parameter registry | Structural separation of invocation inputs vs model weights |
| `trainable` on `Parameter`, derive `requires_grad` at bind | ADR 0006; supports LoRA frozen base |
| `build_graph` factory form | Parameters created inside active composition context |
| Identity lowering for `linear_matmul` | Proves lowering path without fused kernels |
| Weight-only `Linear` first | Smallest proof; bias parameter port follows |

---

## What this unlocks next

After this plan lands:

1. **Phase B — Real modules:** `LayerNorm`, `RMSNorm` affine params + reduction primitives
2. **Phase C — Module-aware lowering:** fused linear, region matching on provenance
3. **Phase D — Estimation:** static weight memory from `graph.parameters`, optimizer
   state from `trainable`, module-attributed reports via provenance + parameter ports

---

## Checklist

- [x] `ValueMetadata` renamed; `TensorMetadata` removed
- [X] `CONTEXT.md` glossary and two-plane diagram
- [X] `GraphParameter` + fixed `context.parameter()`
- [X] `Operation.parameter_ports` + extended validation
- [X] `PortRef.role = "parameter"`
- [X] `StructuralOperation.parameter_ports` bound at graph build
- [X] `apply()` resolves parameter metadata and passes to `infer_result()`
- [X] `LinearMatMul` operation + `linear_matmul()` wrapper
- [X] `Linear` module (weight-only) + `build_graph` factory
- [X] `build_estimation_context()` includes parameter ports
- [X] `linear_matmul` registered in default lowering registry
- [X] `tests/test_parameters.py` passes
- [X] Full test suite passes
