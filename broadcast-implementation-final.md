# Broadcasting — Final Implementation Plan

Add full broadcasting support to Zepto: forward shape rules (already mostly in place
for elementwise binaries), backward reduction FLOPs, backward gradient memory via
**structural auxiliary ports**, and mechanical lowering. Every shipped `Operation`
is updated to the new contract.

Derived from the broadcasting investigation, Option A (Operation owns default FLOPs +
memory declarations; lowering is purely mechanical), and VRAM plan §3 (auxiliary
ports). Lowering already exists (`lower()`, `IdentityImplementation`,
`LoweredOperationValidator`).

---

## Goals

1. **Forward**: NumPy-style broadcasting for elementwise binaries (done) and batch
   broadcasting for `MatMul` (not done). See §3.2 and §5.2 for `_batch_metadata` /
   `_broadcast_batch` semantics.
2. **Backward FLOPs**: charge broadcast reduction (`numel(output) - numel(operand)`)
   and correct VJP costs on every affected op.
3. **Backward memory**: default unfused model — materialized reduced gradients and,
   when required, output-shaped (or batched) unreduced VJP temporaries with explicit
   `ALLOCATE` / `RELEASE` / `PERSIST` pairing.
4. **Contract clarity**: auxiliary ports on `Operation`, `active_auxiliary_ports`
   selection (parallel to `saved_for_backward`), backward events on
   `Operation.resource_events()`.
5. **Mechanical lowering**: remap auxiliary port names and `output:N` to lowered
   tensor ids — no broadcast semantics in lowering.
6. **Documentation and tests** aligned with the above.

---

## Design principles

| Principle | Meaning |
|-----------|---------|
| **Option A** | `Operation` declares default theoretical FLOPs **and** resource events. |
| **Structural auxiliary ports** | Backward gradient tensors are internal ports with inferred metadata and graph `TensorId`s — not a separate `grad:*` namespace invented at lowering. |
| **`active_auxiliary_ports`** | Per-invocation subset of declared aux ports that receive storage and events (mirrors `saved_for_backward`). |
| **Explicit lifetimes** | Unreduced temporaries: `ALLOCATE` → … → `RELEASE`. Reduced gradients: `ALLOCATE` → `PERSIST`. No implicit workspace lifetime. |
| **Sum, not average** | Broadcast backward reduces by **sum** (chain rule). |
| **Pass-through VJP** | `Add` / `Subtract` never materialize an unreduced temp; reduction reads upstream gradient directly. |
| **Fusion seam** | Specialized lowering implementations may omit aux ports / events (ReLU mask, folded MatMul batch). Structural `Operation` keeps the unfused default. |
| **Phase gating** | Backward aux metadata, `active_auxiliary_ports`, and backward events apply when `EstimationContext.phase == "backward"` (or when lowering runs with `InvocationContext.phase == "backward"`). |

---

## Architecture

```text
Operation (contract)
  auxiliary_ports()              — allowed internal port names
  infer_auxiliary_outputs()      — metadata per port (fixed arity)
  active_auxiliary_ports()       — live subset this invocation
  backward_flops()               — VJP + broadcast reduction FLOPs
  resource_events()              — forward allocate(output:N) + backward aux events

        │
        ▼ infer_result() / StructuralGraphBuilder.add_operation()
StructuralOperation
  auxiliary_ports, auxiliary_tensors (TensorId per *active* aux)
  result.active_auxiliary_ports

        │
        ▼ lower() / IdentityImplementation
LoweredOperation
  auxiliary_tensors  — lowered ids for active structural aux
  resource_events    — remapped port names → t… / op… ids
```

---

## Part 1 — Contract records

### 1.1 `src/zepto/core/operation/records.py`

**Extend `OperationResult`:**

```python
@dataclass(frozen=True, slots=True)
class OperationResult:
    outputs: tuple[TensorMetadata, ...]
    auxiliary_outputs: tuple[TensorMetadata, ...] = ()
    aliases: tuple[AliasSpec | None, ...] = ()
    auxiliary_aliases: tuple[AliasSpec | None, ...] = ()
    saved_for_backward: tuple[str, ...] = ()
    active_auxiliary_ports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.aliases and len(self.aliases) != len(self.outputs):
            raise OperationError("Alias declarations must match output arity")
        if self.auxiliary_aliases and len(self.auxiliary_aliases) != len(
            self.auxiliary_outputs
        ):
            raise OperationError(
                "Auxiliary alias declarations must match auxiliary arity"
            )
        if len(self.active_auxiliary_ports) != len(set(self.active_auxiliary_ports)):
            raise OperationError("active_auxiliary_ports must be unique")
```

**Extend `BackwardSpec` docstring** — document gradient-shape invariant:

> For every port name in `gradient_outputs`, the produced gradient has the same
> shape as the forward tensor bound to that port. Broadcast operands imply
> sum-reduction FLOPs (`backward_flops`) and gradient aux ports (`auxiliary_ports`).

---

### 1.2 `src/zepto/core/operation/base.py`

**Add to `SemanticOperation`:**

```python
def auxiliary_ports(self) -> tuple[PortSpec, ...]:
    """Return internal tensor ports not exposed as public functional outputs."""
    return ()

def infer_auxiliary_outputs(
    self,
    inputs: tuple[TensorMetadata, ...],
    outputs: tuple[TensorMetadata, ...],
) -> tuple[TensorMetadata, ...]:
    """Return auxiliary metadata in ``auxiliary_ports`` order."""
    return ()

def active_auxiliary_ports(
    self,
    inputs: tuple[TensorMetadata, ...],
    outputs: tuple[TensorMetadata, ...],
) -> tuple[str, ...]:
    """Return the subset of auxiliary ports live for this invocation."""
    return ()

def auxiliary_aliases(self) -> tuple[AliasSpec | None, ...]:
    """Return per-auxiliary alias declarations (default: fresh storage)."""
    return ()
```

**Update `Operation.infer_result()`:**

```python
auxiliary_outputs = self.infer_auxiliary_outputs(inputs, outputs)
result = OperationResult(
    outputs=outputs,
    auxiliary_outputs=auxiliary_outputs,
    aliases=self.output_aliases(),
    auxiliary_aliases=self.auxiliary_aliases(),
    saved_for_backward=self.saved_for_backward(inputs, outputs),
    active_auxiliary_ports=self.active_auxiliary_ports(inputs, outputs),
)
```

Validate `len(auxiliary_outputs) == len(self.auxiliary_ports())` in
`InvocationValidator`.

---

### 1.3 `src/zepto/core/operation/structural.py`

```python
@dataclass(frozen=True, slots=True)
class StructuralOperation:
    ...
    output_tensors: tuple[TensorId, ...]
    auxiliary_ports: tuple[PortSpec, ...] = ()
    auxiliary_tensors: tuple[TensorId, ...] = ()  # parallel to *active* aux only
    ...
```

Store `auxiliary_ports` from the declaration and `auxiliary_tensors` for each
name in `result.active_auxiliary_ports` (in auxiliary-port declaration order among
active ports, or maintain a full tuple aligned with all auxiliary_ports with
`Optional` — **recommended: full tuple aligned with `auxiliary_ports`, `None`
placeholder omitted — use parallel tuples**:

- `auxiliary_tensors: tuple[TensorId, ...]` — one id per `auxiliary_ports` entry;
  for inactive ports, use a sentinel or omit from graph — **recommended approach**:
  only append TensorIds for active ports, and keep
  `result.active_auxiliary_ports` for resolution. `StructuralOperation` stores:

```python
auxiliary_ports: tuple[PortSpec, ...]
auxiliary_tensors: Mapping[str, TensorId]  # port name → id, active only
```

---

### 1.4 `src/zepto/core/operation/validation.py`

**`DeclarationValidator.validate_ports`:**

- Include `auxiliary_ports()` in the port name uniqueness check (input + output +
  auxiliary must be disjoint and unique).

**`DeclarationValidator.validate_backward`:**

- Extend `known` ports to include auxiliary port names.

**New `InvocationValidator.validate_auxiliary_state`:**

```python
def validate_auxiliary_state(self, operation, inputs, result) -> None:
    aux_ports = operation.auxiliary_ports()
    if len(result.auxiliary_outputs) != len(aux_ports):
        raise OperationError(
            f"{operation.family!r} inferred {len(result.auxiliary_outputs)} "
            f"auxiliary outputs, declares {len(aux_ports)}"
        )
    allowed = {port.name for port in aux_ports}
    if not set(result.active_auxiliary_ports).issubset(allowed):
        raise OperationError("Result activates an undeclared auxiliary port")
    # Gradient-shape invariant for GRADIENT-role aux matching gradient_outputs
    ...
```

Call from `validate_result`.

---

## Part 2 — Graph builder

### 2.1 `src/zepto/core/graph.py` — `add_operation`

After allocating public outputs, allocate auxiliary tensors for each name in
`result.active_auxiliary_ports`:

```python
auxiliary_ports = operation.auxiliary_ports()
aux_by_name = {port.name: port for port in auxiliary_ports}
auxiliary_tensors: dict[str, TensorId] = {}

for port_name in result.active_auxiliary_ports:
    port = aux_by_name[port_name]
    index = next(i for i, p in enumerate(auxiliary_ports) if p.name == port_name)
    metadata = result.auxiliary_outputs[index]
    alias = (
        result.auxiliary_aliases[index]
        if result.auxiliary_aliases
        else None
    )
    # storage: VIEW alias or fresh StorageId (same logic as public outputs)
    aux_id = TensorId(...)
    auxiliary_tensors[port_name] = aux_id
    self._tensors[aux_id] = Tensor(
        aux_id,
        metadata,
        provenance,
        PortRef(operation_id, port.name, "auxiliary"),  # extend PortRef role
        (),
        storage_id,
    )
```

Extend `PortRef` / validation if `"auxiliary"` is not yet a valid role (see
`src/zepto/core/ports.py`).

Pass `auxiliary_ports` and `auxiliary_tensors` into `StructuralOperation`.

Functional composition continues to return **only** public output tensor ids.

---

### 2.2 `src/zepto/core/graph_validation.py`

- Validate every active auxiliary tensor belongs to the graph.
- Validate auxiliary VIEW aliases preserve element count (same as public outputs).

---

## Part 3 — Shared operation helpers

### 3.1 `src/zepto/core/operation/helpers.py`

**FLOP helper (existing plan):**

```python
def broadcast_reduction_flops(
    operand: TensorMetadata, output: TensorMetadata
) -> int:
    return numel(output) - numel(operand)
```

**Aux port name constants (broadcast binaries):**

```python
GRAD_LEFT = "grad_left"
GRAD_RIGHT = "grad_right"
GRAD_LEFT_UNREDUCED = "grad_left_unreduced"
GRAD_RIGHT_UNREDUCED = "grad_right_unreduced"
```

**Metadata builders:**

```python
def reduced_gradient_metadata(operand: TensorMetadata) -> TensorMetadata: ...
def unreduced_gradient_metadata(output: TensorMetadata) -> TensorMetadata: ...
def unreduced_gradient_metadata_matmul(
    output: TensorMetadata, operand: TensorMetadata
) -> TensorMetadata:
    """Shape (*output.shape[:-2], *operand.shape[-2:])."""
    ...
```

**Binary aux inference + activation:**

```python
def binary_auxiliary_ports() -> tuple[PortSpec, ...]:
    return (
        PortSpec(GRAD_LEFT),
        PortSpec(GRAD_RIGHT),
        PortSpec(GRAD_LEFT_UNREDUCED),
        PortSpec(GRAD_RIGHT_UNREDUCED),
    )

def infer_binary_auxiliary_outputs(
    left, right, output, *, materializes_vjp: bool
) -> tuple[TensorMetadata, ...]:
    """Four metadata entries in port order; shapes defined even if later inactive."""

def active_binary_auxiliary_ports(
    left, right, output, *, materializes_vjp: bool
) -> tuple[str, ...]:
    """Select live ports (see Part 5)."""
```

**Event helpers:**

```python
def allocate(result: OperationResult) -> tuple[ResourceEvent, ...]:
    """Unchanged: ALLOCATE output:0, …"""

def backward_gradient_port_events(
    port_name: str,
    *,
    unreduced_port: str | None,
) -> tuple[ResourceEvent, ...]:
    """ALLOCATE unreduced? → ALLOCATE port → RELEASE unreduced? → PERSIST port."""

def emit_active_auxiliary_events(
    result: OperationResult,
    event_builder: Callable[[str], tuple[ResourceEvent, ...]],
) -> tuple[ResourceEvent, ...]:
    """Call event_builder(port_name) for each name in result.active_auxiliary_ports."""
```

### 3.2 MatMul batch helpers — `matmul.py` (module-level)

These helpers split **batch prefix** from **matrix block** so batch broadcasting
reuses the existing elementwise `broadcast_metadata` machinery without touching
the contracting dimensions.

**Shape convention (PyTorch / NumPy `matmul`, rank ≥ 2):**

```text
operand.shape == (*batch_prefix, M, K)   # left
operand.shape == (*batch_prefix, K, N)   # right   (K must match)
output.shape  == (*batch_prefix, M, N)   # after batch broadcast

*batch_prefix  — all dimensions before the final two (may differ in rank across operands)
(M, K) / (K, N) — fixed matrix blocks; never broadcast against each other
```

```python
def _batch_metadata(value: TensorMetadata) -> TensorMetadata:
    """Return metadata for the leading batch prefix only.

    For an operand ``(*batch, M, K)`` this is metadata with shape ``batch`` —
    i.e. ``value.shape[:-2]``. The final two dimensions are the matrix block
    and are handled separately by contraction rules in ``infer_outputs``.

    Examples:
        (32, 128, 512)  → batch shape (32,)
        (512, 64)       → batch shape ()      # rank-2 weight: no leading batch
        (8, 1, 4, 4)    → batch shape (8, 1)

    Wrapping in ``TensorMetadata`` (instead of returning a bare ``tuple[int, ...]``)
    lets us pass the batch prefix into ``broadcast_metadata`` unchanged — that
    function already implements NumPy-style rank promotion and singleton
    expansion for elementwise binaries.
    """
    return TensorMetadata(value.shape[:-2])


def _broadcast_batch(
    left: TensorMetadata, right: TensorMetadata, *, family: str
) -> tuple[int, ...]:
    """Broadcast the leading batch dimensions of two matmul operands.

    Steps:
    1. Strip matrix blocks: ``left_batch = left.shape[:-2]``,
       ``right_batch = right.shape[:-2]``.
    2. Delegate to ``broadcast_metadata((left_batch_meta, right_batch_meta))`` —
       same rules as ``Add`` / ``Multiply`` (rank padding, singleton ``1``
       expands, incompatible sizes raise ``ValueError``).
    3. Return the resulting batch shape; append ``(M, N)`` in ``infer_outputs``.

    Why not call ``broadcast_metadata(left, right)`` on full shapes?
    That would try to broadcast dimension 512 against 64 across the matrix
    block — wrong. Batch broadcast and matrix contraction are orthogonal.

    Why not duplicate broadcast logic inline in ``MatMul``?
    One implementation of NumPy rules (``broadcast_metadata``) stays authoritative;
    ``MatMul`` only scopes it to ``[:-2]``.
    """
    return broadcast_metadata(
        (_batch_metadata(left), _batch_metadata(right)),
        family=family,
    ).shape


def _batch_reduction_flops(
    operand: TensorMetadata, output: TensorMetadata
) -> int:
    """Additions to sum a batched operand gradient over broadcast batch axes.

    Unfused batched backward materializes a gradient of shape
    ``(*output.shape[:-2], *operand.shape[-2:])`` and sums over axes where the
    operand's batch prefix was broadcast up to the output's batch prefix.
    Returns ``numel(unreduced) - numel(operand)``; 0 when batch prefixes already match.
    """
    output_batch = numel(_batch_metadata(output))
    unreduced = output_batch * operand.shape[-2] * operand.shape[-1]
    return unreduced - numel(operand)
```

Import ``broadcast_metadata`` from ``.helpers`` inside ``matmul.py``.

---

## Part 4 — Lowering (mechanical only)

### 4.1 `src/zepto/core/lowering/helpers.py`

**Extend `remap_events`:**

```python
for port_name, tensor_id in structural.auxiliary_tensors.items():
    port_targets[port_name] = tensor_map[tensor_id]
```

**Extend `port_tensor_id`** to accept `role="auxiliary"`.

**Extend `ensure_lowered_tensor` loop in identity impl** — materialize every
auxiliary structural tensor before remapping events.

### 4.2 `src/zepto/core/lowering/implementations/identity.py`

```python
for tensor_id in structural.input_tensors:
    ensure_lowered_tensor(...)
for tensor_id in structural.output_tensors:
    ensure_lowered_tensor(...)
for tensor_id in structural.auxiliary_tensors.values():
    ensure_lowered_tensor(...)

events = op.resource_events(estimation, result)
remapped = remap_events(events, structural, tensor_map)
with_saves = append_save_events(remapped, structural, tensor_map)

auxiliary_ids = tuple(
    tensor_map[tid] for tid in structural.auxiliary_tensors.values()
)

return LoweredOperation(
    ...
    auxiliary_tensors=auxiliary_ids,
    resource_events=with_saves,
    ...
)
```

**Remove** any broadcast-specific logic from lowering (no
`append_broadcast_backward_events`, no `broadcast_materializes_vjp` in identity).

### 4.3 `src/zepto/core/lowering/validation.py`

Optional: validate each active aux lowered tensor's metadata shape against primal
port for `GRADIENT` role.

### 4.4 ReLU (`maximum/relu-mask`)

Longer term: express mask as `Maximum.auxiliary_ports = (PortSpec("mask"),)` and
move mask events to `Operation.resource_events()`; until then ReLU impl may remain
a specialized override that **replaces** the default event list.

---

## Part 5 — Per-operation updates

Legend:

- **VJP mat?** — materializes output-shaped (or batched) unreduced temp when operand
  is broadcast.
- **Aux** — uses four binary aux ports unless noted.

### 5.1 Elementwise binaries

| Op | `backward_flops` | VJP mat? | Aux activation rules |
|----|------------------|----------|----------------------|
| `Add` | reduction only per operand | No | `grad_*` when `requires_grad` and operand shape ≠ output; never `*_unreduced` |
| `Subtract` | reduction + `numel(right)` negation | No | same as Add |
| `Multiply` | `numel(output)` + reduction per operand | Yes | `grad_left_unreduced` if left broadcast; `grad_right_unreduced` if right broadcast |
| `Divide` | quotient rule + reduction | Yes | same unreduced rules as Multiply |
| `Maximum` | masked multiply + reduction | Yes | same; ReLU path unchanged at lowering |
| `Minimum` | masked multiply + reduction | Yes | same |

**Shared pattern for `Add` (representative):**

```python
@property
def auxiliary_ports(self):
    return binary_auxiliary_ports()

def infer_auxiliary_outputs(self, inputs, outputs):
    left, right = inputs
    (output,) = outputs
    return infer_binary_auxiliary_outputs(
        left, right, output, materializes_vjp=False
    )

def active_auxiliary_ports(self, inputs, outputs):
    left, right = inputs
    (output,) = outputs
    return active_binary_auxiliary_ports(
        left, right, output, materializes_vjp=False
    )

def backward_flops(self, context):
    ...

def resource_events(self, context, result):
    events = list(allocate(result))
    if context.phase != "backward":
        return tuple(events)
    for port_name in result.active_auxiliary_ports:
        events.extend(
            (
                ResourceEvent(ALLOCATE, port_name, phase="backward"),
                ResourceEvent(PERSIST, port_name, phase="backward"),
            )
        )
    return tuple(events)
```

**`Multiply` — backward events for broadcast right operand:**

```python
if GRAD_RIGHT in result.active_auxiliary_ports:
    if GRAD_RIGHT_UNREDUCED in result.active_auxiliary_ports:
        events.extend(
            backward_gradient_port_events(
                GRAD_RIGHT, unreduced_port=GRAD_RIGHT_UNREDUCED
            )
        )
    else:
        events.extend(persist_only_events(GRAD_RIGHT))
```

**Docstring fix:** `Add` — "NumPy-style broadcast semantics" (not "equal-rank").

---

### 5.2 `MatMul` — `src/zepto/core/operation/matmul.py`

See §3.2 for helper definitions. This section explains **how they fit together**
and **why** the current equal-rank / equal-batch checks are removed.

#### 5.2.1 Shape decomposition (why `[:-2]`)

Every operand with rank ≥ 2 is treated as two independent pieces:

| Piece | Slice | Role |
|-------|-------|------|
| Batch prefix | `shape[:-2]` | Broadcast NumPy-style between operands (via `_broadcast_batch`) |
| Matrix block | `shape[-2:]` | Contract: `left[..., M, K] @ right[..., K, N] → [..., M, N]` |

The batch prefix and matrix block must **never** be confused:

- `(32, 128, 512)` is batch `(32,)` + matrix `(128, 512)` — **not** batch `(32, 128)`.
- The middle dimension `128` is the **M** row count of the left matrix, not a batch axis.
- PyTorch `matmul((32,128,512), (512,64)) → (32,128,64)` follows exactly this split.

Reject rank `< 2` (no matrix block). Do **not** require equal rank between operands:
rank promotion applies only to the batch prefix inside `broadcast_metadata`.

#### 5.2.2 `_batch_metadata()` — strip the matrix block

Returns `TensorMetadata` whose shape is **only** the batch prefix `value.shape[:-2]`.

It exists so `_broadcast_batch` can call the existing `broadcast_metadata` helper
without copying rank-padding / singleton-expansion logic into `MatMul`.

| Operand shape | `_batch_metadata(...).shape` | Notes |
|---------------|------------------------------|-------|
| `(32, 128, 512)` | `(32,)` | Activations `(B, S, D)` |
| `(512, 64)` | `()` | Rank-2 weight: empty batch prefix |
| `(8, 1, 4, 4)` | `(8, 1)` | Two batch axes before `(4, 4)` matrix |

An empty batch shape `()` is valid (rank-2 weight). `broadcast_metadata` promotes
it with leading `1`s until ranks match, same as elementwise `(512,) + (32,128,512)`.

#### 5.2.3 `_broadcast_batch()` — batch-only NumPy broadcast

```python
batch = _broadcast_batch(left, right, family=self.family)
```

Internally:

```text
left_batch  = left.shape[:-2]     e.g. (32,)
right_batch = right.shape[:-2]    e.g. () for weight (512, 64)

broadcast_metadata wraps both → e.g. (32,) after promoting () → (1,) and broadcasting
```

**Worked example A — linear layer (most common training case):**

```text
left:  (32, 128, 512)   activations
right: (512, 64)        weight

left batch:   (32,)
right batch:  ()         promoted to (1,) → broadcast → (32,)

matrix:  left[-2:] = (128, 512)  → M=128, K=512
         right[-2:] = (512, 64)  → K=512, N=64   ✓ contracting K matches

output shape: (*batch, M, N) = (32, 128, 64)
```

**Worked example B — two batch axes with singleton broadcast:**

```text
left:  (8, 1, 4, 4)
right: (1, 5, 4, 4)

left batch:   (8, 1)
right batch:  (1, 5)
broadcast_metadata → (8, 5)

matrix: both (4, 4); output shape (8, 5, 4, 4)
```

**Worked example C — rejection (unchanged from elementwise):**

```text
left batch:  (2,)
right batch: (3,)   on matrix-compatible operands → ValueError from broadcast_metadata
```

#### 5.2.4 Full `infer_outputs` (replaces equal-rank / equal-batch checks)

```python
def infer_outputs(
    self, inputs: tuple[TensorMetadata, ...]
) -> tuple[TensorMetadata, ...]:
    left, right = inputs
    if len(left.shape) < 2 or len(right.shape) < 2:
        raise ValueError(
            f"MatMul expects rank >= 2, got {left.shape} and {right.shape}"
        )
    batch = _broadcast_batch(left, right, family=self.family)
    m, left_k = left.shape[-2:]
    right_k, n = right.shape[-2:]
    if left_k != right_k:
        raise ValueError(
            f"incompatible MatMul contracting dimensions: "
            f"{left.shape} @ {right.shape}"
        )
    requires_grad = any(value.requires_grad for value in inputs)
    return (
        TensorMetadata(
            shape=(*batch, m, n),
            semantic_type=left.semantic_type,
            requires_grad=requires_grad,
        ),
    )
```

**Why output batch comes from `_broadcast_batch`, not from `left` alone:** after
broadcast, the output batch prefix is the **union** of both operands' batch axes
(example B: `(8,1)` and `(1,5)` → `(8,5)`). Using only `left.shape[:-2]` would
under-count GEMMs and mis-size backward temporaries when the right operand carries
the larger batch prefix.

#### 5.2.5 FLOPs — why `_batch_metadata(output)` is the batch factor

```python
def forward_flops(self, context: EstimationContext) -> int:
    left = context.metadata_for("left")
    right = context.metadata_for("right")
    output = context.metadata_for("output")
    batch = numel(_batch_metadata(output))   # not numel(_batch_metadata(left))
    return 2 * batch * left.shape[-2] * left.shape[-1] * right.shape[-1]
```

`numel(_batch_metadata(output))` counts how many independent `(M, N)` matrix
products run — i.e. the **broadcast** batch volume. Example A: `numel((32,)) = 32`
GEMMs, each `2 * 128 * 512 * 64` FLOPs.

Using `left.shape[:-2]` alone would be wrong in example B (`numel((8,1)) = 8` but
output batch is `(8,5)` → 40 GEMMs).

#### 5.2.6 Backward FLOPs and `_batch_reduction_flops`

Each requested operand gradient costs one forward-sized GEMM at **output** batch
size, plus `_batch_reduction_flops(operand, output)` when that operand's batch
prefix was broadcast:

```python
batch = numel(_batch_metadata(output))
if right.requires_grad:
    flop += 2 * batch * m * n * k
    flop += _batch_reduction_flops(right, output)
```

Example A, `dRight`: unfused backward forms `(32, 512, 64)` then sums to
`(512, 64)` → `_batch_reduction_flops = 32*512*64 - 512*64`.

Folded batch-into-contraction lowering may omit the unreduced temp (deferred
`matmul/folded-batch` implementation); structural default charges unfused cost.

#### 5.2.7 Auxiliary ports and backward memory

Same four port names as elementwise binaries (§3.1). MatMul-specific rules:

**`infer_auxiliary_outputs`:** unreduced gradient shape
`(*output.shape[:-2], *operand.shape[-2:])` when batch was broadcast on that
operand; reduced gradient shape matches `operand.shape`.

**`active_auxiliary_ports`:**

- `grad_left` / `grad_right` when respective `requires_grad`.
- `grad_left_unreduced` when `left.requires_grad` and
  `left.shape[:-2] != output.shape[:-2]`.
- `grad_right_unreduced` when `right.requires_grad` and
  `right.shape[:-2] != output.shape[:-2]`.

Example A: only `grad_right_unreduced` (+ `grad_right`, `grad_left` if modeling
full operand grad storage) — left batch `(32,)` matches output batch `(32,)`.

**`resource_events`:** same event helper pattern as Multiply; phase-gated on
`context.phase == "backward"`.

Update class docstring: leading dimensions broadcast NumPy-style; they no longer
"must currently match".

---

### 5.3 Unary / view / multi-output ops (no cross-operand broadcast)

| Op | `auxiliary_ports` | `backward_flops` | `resource_events` |
|----|-------------------|------------------|-------------------|
| `Identity` | `()` | 0 | `ALIAS output:0` (unchanged) |
| `Reshape` | `()` | 0 | `ALIAS output:0` |
| `Transpose` | `()` | 0 | `ALIAS output:0` |
| `SquareRoot` | `()` or `(PortSpec("grad_input"),)` if modeling input grad storage | existing + 0 reduction | forward `allocate`; optional single grad aux if desired |
| `Split` | `()` for this milestone | existing | `allocate` per output |

Split backward is concatenation, not elementwise broadcast between operands —
out of scope for binary aux ports. Document in op docstring.

---

### 5.4 Files to touch (operations)

| File | Changes |
|------|---------|
| `add.py` | aux ports, `backward_flops`, `resource_events`, docstring |
| `subtract.py` | same |
| `multiply.py` | same + VJP unreduced |
| `divide.py` | same + VJP unreduced |
| `maximum.py` | same + VJP unreduced |
| `minimum.py` | same + VJP unreduced |
| `matmul.py` | batch infer/FLOPs + aux + events |
| `identity.py` | confirm empty aux; no backward events |
| `reshape.py` | confirm empty aux |
| `transpose.py` | confirm empty aux |
| `square_root.py` | confirm empty aux (unless grad_input added) |
| `split.py` | confirm empty aux |

---

## Part 6 — Ports and composition

### 6.1 `src/zepto/core/ports.py`

- Allow `PortRef.role == "auxiliary"` (or extend `ValueKind` if aux ports use a
  distinct value kind — align with existing port model).
- Update any role guards in graph validation.

### 6.2 `src/zepto/core/composition.py`

- No public API change: functional wrappers still return only public outputs.
- Ensure `add_operation` path uses updated builder signature if needed.

---

## Part 7 — Documentation

### 7.1 `docs/primitive-backward.md`

Add **Broadcasting** section:

- Gradient-shape invariant (per-port).
- Sum-not-average rule.
- Reduction FLOP formula: `numel(output) - numel(operand)`.
- Default unfused memory: auxiliary ports, unreduced temp + reduced grad lifetimes.
- MatMul batch broadcast and folded-GEMM lowering optimization note.

Update per-op sections (Add/Subtract already mention sum — extend to all binaries
and MatMul).

### 7.2 `CONTEXT.md`

Optional domain-modeling follow-up: define **unreduced gradient** as a named term
linked to `TensorRole.WORKSPACE` aux ports.

### 7.3 `broadcast-implementation-final.md`

This file is the implementation source of truth; supersede
`broadcasting-implementation.md` (archive or delete after implementation).

### 7.4 ADR pointer

Note in PR description: aligns with ADR-0001 (structural vs lowered), ADR-0003
(resource events), VRAM plan §3 (auxiliary ports).

---

## Part 8 — Tests

### 8.1 `tests/test_operation_contract.py`

**Helpers / FLOPs:**

```python
class TestBroadcastReductionFlops: ...
class TestBroadcastBackwardFlops:
    # x (32,128,512), bias (512,)
    # Add: reduction only
    # Subtract: reduction + numel(bias) negation
    # Multiply: VJP + reduction on broadcast operand
class TestMatMulBatchBroadcast:
    # infer (32,128,512) @ (512,64) → (32,128,64)
    # singleton batch dims
    # incompatible batch / contraction rejected
    # backward_flops includes batch reduction
```

**Contract / aux ports:**

```python
class TestAuxiliaryPorts:
    def test_multiply_declares_four_aux_ports(self): ...
    def test_active_aux_ports_omit_unreduced_without_broadcast(self): ...
    def test_active_aux_includes_unreduced_for_multiply_bias(self): ...
    def test_infer_result_validates_auxiliary_arity(self): ...
    def test_identity_has_no_aux_ports(self): ...
```

**Validation:**

```python
def test_gradient_aux_metadata_matches_operand_shape(): ...
def test_active_aux_must_be_declared_port(): ...
```

### 8.2 `tests/test_structural_graph.py`

```python
def test_multiply_allocates_active_aux_tensors_only(): ...
def test_inactive_aux_ports_have_no_tensor_id(): ...
def test_auxiliary_view_alias_preserves_storage_when_declared(): ...
```

### 8.3 `tests/test_lowering.py`

```python
class TestBroadcastBackwardLowering:
    def test_multiply_backward_remaps_aux_port_events(self): ...
    def test_multiply_forward_has_no_backward_events(self): ...
    def test_all_backward_event_targets_are_known_tensors(self): ...
    def test_unreduced_allocate_release_order(self): ...
    def test_matmul_weight_backward_aux_shapes(self): ...
```

Use `reference_context(phase="backward")` for backward cases.

### 8.4 Regression

- Update tests expecting old `Add` asymmetric backward FLOPs.
- Update tests expecting `MatMul` equal-batch rejection.
- `pytest tests/ -q` green after each execution step.

---

## Part 9 — Execution order

| Step | Work | Depends on |
|------|------|------------|
| 1 | §1 records + §1.2 base hooks + validators (aux arity, active subset) | — |
| 2 | §3 helpers (`broadcast_reduction_flops`, binary aux helpers, event helpers) | 1 |
| 3 | §2 graph builder + §6 ports (`auxiliary_tensors`, PortRef role) | 1 |
| 4 | §4 lowering (`remap_events`, identity materialize aux) | 3 |
| 5 | §5.1 `Add` + `Subtract` (simplest aux + FLOPs) | 2, 4 |
| 6 | §5.1 remaining binaries (`Multiply` … `Minimum`) | 5 |
| 7 | §5.2 `MatMul` batch + aux | 2, 4 |
| 8 | §5.3 confirm unary/view ops (empty aux) | 1 |
| 9 | §7 documentation | 5–7 |
| 10 | §8 tests alongside steps 5–7 | same |

---

## Part 10 — Worked example (acceptance)

**Graph:** `Multiply(x, bias)`, `x: (32,128,512)`, `bias: (512,)`, both
`requires_grad=True`.

**After `infer_result`:**

| Field | Value |
|-------|--------|
| `outputs[0].shape` | `(32, 128, 512)` |
| `active_auxiliary_ports` | `("grad_left", "grad_right", "grad_right_unreduced")` or omit `grad_left` if same-shape pass-through policy |
| `auxiliary_outputs["grad_right"].shape` | `(512,)` |
| `auxiliary_outputs["grad_right_unreduced"].shape` | `(32, 128, 512)` |

**`resource_events` (phase=backward) — order:**

```text
ALLOCATE output:0                    # forward; remapped to t_out
ALLOCATE grad_right_unreduced
ALLOCATE grad_right
RELEASE  grad_right_unreduced
PERSIST  grad_right
ALLOCATE grad_left                   # if active
PERSIST  grad_left
```

**`backward_flops`:** `numel(output)` for left VJP + `numel(output) + reduction` for
right.

**After `lower(..., phase="backward")`:** all event targets are lowered tensor ids;
`LoweredOperationValidator` passes; auxiliary tensors listed on the lowered op.

---

## Part 11 — Deferred work

| Item | Notes |
|------|-------|
| `account_memory()` | Replays lowered events; bytes from aux `TensorMetadata` + policy |
| `Sum` / `Reduce` primitive | Compositional backward; adjoint to broadcast |
| `Broadcast` lowering artifact | Normalization only, never user-facing |
| ReLU mask as structural aux | Refactor `maximum/relu-mask` to use `auxiliary_ports` |
| Fused implementations | `matmul/folded-batch`, fused broadcast-reduce — omit unreduced aux |
| Split concat backward aux | Separate design if grad storage is modeled explicitly |

---

## Summary

Broadcasting support is delivered by:

1. **Structural auxiliary ports** with **`active_auxiliary_ports`** selection.
2. **`Operation.backward_flops`** for reduction + VJP FLOPs.
3. **`Operation.resource_events`** for forward output allocation and backward aux
   lifetimes (explicit `ALLOCATE` / `RELEASE` / `PERSIST`).
4. **Graph builder** allocating aux `TensorId`s for active ports only.
5. **Mechanical lowering** remapping port names — no broadcast logic in lowering.
6. **Updating all shipped operations** per Part 5 and tests/docs in Parts 7–8.

This is the single implementation plan for the broadcasting workstream.

