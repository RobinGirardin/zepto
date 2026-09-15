# Estimation Feature — Implementation Plan (Approach A + Horizon)

**Status:** planning artifact — no code landed yet  
**Last updated:** 2026-09-14  
**Audience:** agents and humans implementing Zepto cost estimation in phases

This document is the authoritative build plan for the estimation API. It supersedes earlier chat summaries. Read ADR-0003, ADR-0006, and `CONTEXT.md` before implementing any phase.

---

## 1. Goals and non-goals

### Goals

- **Approach A transparency:** every pipeline stage is independently callable; `estimate()` is a thin composer only.
- **Single invocation:** `compose → lower → account_memory → account_flops → CostReport`.
- **Horizon:** orchestrates many invocations (prefill, decode, training micro-batches) and produces timeline-reduced reports.
- **Training support:** optimizer state and gradient accumulation are in scope from Phase 2 — not deferred.
- **Correct peak VRAM:** peak live bytes must match Atto-style schedule semantics, not sum-all inventory semantics.

### Non-goals (deferred ergonomics)

- Backend profile presets (`apertus_hf_flash2`, …)
- Graph caching / scenario sweeps
- `estimate_apertus(...)` convenience wrappers
- HTTP/FastAPI service layer

---

## 2. Target call stacks

### 2.1 Single invocation (explicit)

```text
graph   = compose_graph(...)
lowered = lower(graph, ctx)          # includes LoweredParameter records
mem     = account_memory(lowered)    # self-contained — no Graph argument
flops   = account_flops(lowered)
report  = CostReport(memory=mem, flops=flops, context=ctx)
```

Optional helper (not required):

```python
report = combine_cost_report(mem, flops, context=ctx)
```

### 2.2 Horizon (explicit)

```text
spec = HorizonSpec.inference(prefill=8192, decode_steps=128, kv=KVConfig(...))
sim  = simulate_horizon(spec, module_fn, inputs_fn, ctx)
hmem = account_memory(sim)
hflops = account_flops(sim)
report = HorizonCostReport(...)
```

Training example:

```text
spec = HorizonSpec.training(seq_len=2048, micro_batches=4, optimizer=AdamW)
sim  = simulate_horizon(spec, module_fn, inputs_fn, ctx)
```

### 2.3 Facade (thin composer)

```python
report = estimate(graph, ctx)                                              # → CostReport
hcost  = estimate_horizon(spec, module_fn, inputs_fn, ctx)                 # → HorizonCostReport
# escape hatches: return_lowered=True, return_simulation=True
```

---

## 3. Core invariants

| Invariant | Source |
|-----------|--------|
| One `LoweredGraph` = one concrete invocation | ADR-0003 |
| Structural `Graph` is backend-neutral; lowering is variable | ADR-0001 |
| `LoweredGraph` is the authoritative costing artifact and carries parameters | This plan §4 |
| Horizon peak ≠ `max(step.peak)` — persistent weights, KV, optimizer state carry across steps | ADR-0003 |
| One graph per horizon step — prefill and decode re-compose with different shapes | ADR-0003, ADR-0005 |
| `sum_all_bytes` and `peak_live_bytes` are distinct metrics | ADR-0006, Atto §3 |
| Resource events declare allocations; the simulator adds schedule-driven RELEASE | Atto peak model |

---

## 4. Peak vs sum-all — Atto-aligned semantics

### 4.1 The mistake in the first draft

Two errors were bundled in the initial simulator spec:

1. **`PERSIST` on activation gradient ports (`grad_left`, `grad_right`, …).**  
   `persist_only_events()` and the tail of `backward_gradient_port_events()` currently mark reduced grad auxiliary ports with `PERSIST`. That is appropriate for a **sum-all inventory** (“this buffer exists at some point during backward”) but **wrong for peak VRAM** (“how many bytes are live at once?”).

2. **Vague SAVE release.**  
   “Release saved forward activations after backward consumes them” is insufficient. Release is keyed to **`last_use`**: the forward index of the **earliest** backward node whose backward still needs that save.

### 4.2 Three gradient concepts (do not mix)

| Concept | What it is | Peak treatment | Sum-all treatment |
|---------|------------|----------------|-------------------|
| **Forward activation `X`** | Output of forward node N | Live from N's ALLOCATE until schedule-driven RELEASE at `last_use(X)` | Every ALLOCATE counts |
| **Saved-for-backward `X`** | Operand retained via SAVE | Pinned from forward SAVE until `last_use` RELEASE | SAVE marks existence |
| **Input activation grad `gX`** | Gradient flowing *into* operand X at backward node M | Transient wavefront: ALLOCATE at M, RELEASE at **producer node** of X (earlier in topo order), not at M | ALLOCATE counts once |
| **Weight grad accum `gW`** | Persistent optimizer-bound gradient | PERSIST across micro-batches / horizon steps | PERSIST counts |
| **Unreduced temp** | Broadcast VJP workspace | ALLOCATE → RELEASE within same backward node | Both count |

The operation **cannot** emit the RELEASE timestamp for `gX` because it depends on where `X` was produced in the graph. The **simulator** computes `last_use` from the lowered graph topology and injects RELEASE events at the correct schedule point.

### 4.3 Required changes to operation event helpers

**File:** `src/zepto/semantic/operations/helpers.py`

```python
def activation_grad_events(port_name: str) -> tuple[ResourceEvent, ...]:
    """Backward input activation grad: allocate only; simulator schedules release."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, port_name, phase="backward"),
    )


def unreduced_grad_events(unreduced_port: str) -> tuple[ResourceEvent, ...]:
    """Transient workspace: allocate and release within the same backward node."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, unreduced_port, phase="backward"),
        ResourceEvent(ResourceEventKind.RELEASE, unreduced_port, phase="backward"),
    )


def weight_grad_accum_events(port_name: str) -> tuple[ResourceEvent, ...]:
    """Parameter gradient accumulator: survives across backward nodes / horizon steps."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, port_name, phase="backward"),
        ResourceEvent(ResourceEventKind.PERSIST, port_name, phase="backward"),
    )


def backward_gradient_port_events(
    port_name: str,
    *,
    unreduced_port: str | None,
    is_parameter: bool = False,
) -> tuple[ResourceEvent, ...]:
    """Emit backward auxiliary resource events with correct peak semantics."""
    events: list[ResourceEvent] = []
    if unreduced_port is not None:
        events.extend(unreduced_grad_events(unreduced_port))
    if is_parameter:
        events.extend(weight_grad_accum_events(port_name))
    else:
        events.extend(activation_grad_events(port_name))
    return tuple(events)
```

**Migration:**

- Replace `persist_only_events()` call sites with `activation_grad_events()` for activation grads.
- Keep `weight_grad_accum_events()` (rename from `persist_only_events`) only for parameter gradient ports.
- Remove the final `PERSIST` from `backward_gradient_port_events()` for activation ports.
- Update `tests/test_lowering.py::TestBackwardResourceEvents::test_unreduced_allocate_release_order` — remove assertion that `PERSIST` appears for activation grad ports.

**Impact:** touches every operation using `persist_only_events` / `backward_gradient_port_events` (Add, MatMul, Pow, Where, Sin, Cos, Exp, Log, Cast, Concat, Gather, ReduceSum, RepeatKV, EmbeddingLookup, ParameterBias, Subtract, …).

---

## 5. `ResourceEventSimulator` — detailed specification

This is **Phase 1a** — implement before reports, horizon, or Apertus integration.

### 5.1 Architecture: two-pass model

```text
Pass 1 — Event collection (per node, declaration order)
  Bootstrap: parameters, graph inputs, external state ports
  For each LoweredNode in topological order:
    Apply declared resource_events (ALLOCATE, ALIAS, SAVE, PERSIST, RELEASE, WORKSPACE)
    Apply node-scoped workspace auto-release

Pass 2 — Schedule augmentation (graph-wide, Atto-style)
  Build last_use map for every edge/storage slot
  Inject RELEASE for:
    - forward activations at last consumer backward boundary
    - saved-for-backward tensors at last_use(save)
    - activation grads gX at producer node of X
  Recompute peak_live_bytes on augmented timeline
```

Pass 1 alone produces correct **sum-all** and a conservative peak. Pass 2 produces correct **training peak** (e.g. MatMul → ReLU → MatMul chain without ~3× over-count).

### 5.2 Storage model

```python
@dataclass
class StorageSlot:
    storage_id: str
    bytes: int
    live: bool
    refcount: int
    kind: Literal[
        "parameter", "activation", "saved", "workspace",
        "gradient_wavefront", "weight_grad", "state", "persistent_input",
    ]
    pinned: bool          # SAVE or PERSIST
    producer_node: int | None   # forward node index that allocated
    edge_id: str | None
```

Each `LoweredEdge.storage_id` maps to a slot. ALIAS shares storage and increments refcount without adding bytes.

### 5.3 Bootstrap (before node events)

Process in fixed order:

1. **Parameters** (`lowered.parameters`): `ALLOCATE` + `PERSIST` for each `LoweredParameter`. Count in `breakdown.parameters`; live from step 0.

2. **Graph inputs** (`role == INPUT`):
   - `tensor.persistent == True` (causal mask, RoPE tables): `ALLOCATE` + `PERSIST` → `breakdown.persistent_inputs`
   - else: `ALLOCATE` (live until schedule release unless graph output)

3. **External state ports** (`InvocationContext.state` — KV, grad accum, optimizer from horizon):
   - Pre-existing buffers: `ALLOCATE` + `PERSIST` → `breakdown.state`
   - Not released at invocation end unless explicit `RELEASE`

### 5.4 Phase filtering

| `context.phase` | Active events |
|-----------------|---------------|
| `"forward"` | `event.phase == "forward"` |
| `"backward"` | `event.phase == "backward"` |
| `"full"` | both (forward nodes first in topo order, then backward) |

For `phase="full"`, lowering must emit both forward and backward event sets. Today many operations gate backward events on `context.phase != "backward"`. **Phase 1a follow-up:** extend `reference_invocation(phase="full")` lowering path or run two lowering passes merged — pick one approach and document in Phase 1a PR.

**Recommended for Phase 1a:** start with separate `phase="forward"` and `phase="backward"` invocations; add `phase="full"` in Phase 1b once schedule augmentation works on a split forward/backward horizon pair.

### 5.5 Event kind semantics (Pass 1)

| Kind | Action | sum_all | peak (before Pass 2) |
|------|--------|---------|----------------------|
| **ALLOCATE** | Create/mark live slot; bytes from `ResolvedValue` | += bytes | recalc |
| **ALIAS** | Share storage; refcount++ | no change | no change |
| **SAVE** | Pin slot; mark `kind=saved` | no change | no change |
| **PERSIST** | Pin slot; survives invocation | no change | no change |
| **RELEASE** | refcount--; free if 0 and not pinned | no change | recalc |
| **WORKSPACE** | ALLOCATE tagged workspace; auto-release at node end | += bytes | recalc |

**Node-scoped cleanup:** after each node's declared events, release workspace slots allocated during that node unless SAVE/PERSIST/graph-output.

### 5.6 Pass 2 — Schedule-driven RELEASE

```python
@dataclass(frozen=True)
class LastUseRecord:
    edge_id: str
    producer_node_index: int
    last_consumer_node_index: int   # last backward node needing this tensor

def build_last_use_map(lowered: LoweredGraph) -> dict[str, LastUseRecord]:
    """
    For each edge/storage:
    - Forward activations: last node in topo order that reads the edge
      (forward or backward phase).
    - SAVE'd operands: last backward node whose saved_for_backward includes
      the edge (from structural node metadata via lowered.node_map).
    - Activation grads (auxiliary GRAD_* ports): producer_node_index of the
      corresponding forward input edge.
    """
    ...


def augment_timeline_with_scheduled_releases(
    timeline: list[TimelineEvent],
    last_use: dict[str, LastUseRecord],
    lowered: LoweredGraph,
) -> list[TimelineEvent]:
    """
    Insert RELEASE events at:
    1. End of last_consumer_node for forward activations and SAVE'd tensors.
    2. End of producer_node for activation grad wavefronts (gX released where X was produced).
    Never release PERSIST'd parameter / state / weight_grad slots.
    """
    ...
```

**Example: MatMul → ReLU → MatMul (training, phase=full or forward+backward horizon steps)**

```text
Forward:
  N0 MatMul: ALLOCATE Y0
  N1 ReLU:   ALLOCATE Y1, SAVE mask (or input per recipe)
  N2 MatMul: ALLOCATE Y2

Backward (reverse topo):
  N2 MatMul: ALLOCATE gY1 (wavefront), ALLOCATE gW2 (weight accum PERSIST)
  N1 ReLU:   ALLOCATE gY0 (wavefront)
  N0 MatMul: ALLOCATE gX, gW0 (weight accum PERSIST)

Scheduled releases (Pass 2):
  gY1 released at end of N1 backward (producer of Y1 into N2)
  gY0 released at end of N0 backward
  Y0 released after last use
  SAVE'd mask released after N1 backward completes
  Weight grad accum gW* remain PERSIST until optimizer step / horizon boundary
```

Peak ≈ |weights| + |max forward wavefront| + |one backward wavefront| — **not** ~3×|X| from PERSIST on every grad port.

### 5.7 Outputs

```python
@dataclass(frozen=True)
class SimulationResult:
    sum_all_bytes: int
    peak_live_bytes: int          # after Pass 2 augmentation
    peak_live_bytes_naive: int    # Pass 1 only; useful for regression/debug
    breakdown: MemoryBreakdown
    timeline: tuple[TimelineEvent, ...]
    by_module: tuple[AttributionSlice, ...]
    by_region: tuple[AttributionSlice, ...]
    by_implementation: tuple[AttributionSlice, ...]
```

### 5.8 Breakdown buckets

| Bucket | Source |
|--------|--------|
| `parameters` | Bootstrap parameter slots |
| `persistent_inputs` | Input edges with `persistent=True` |
| `activations` | Forward ALLOCATE'd intermediates |
| `saved_for_backward` | SAVE'd slots during forward |
| `workspace` | WORKSPACE + unreduced temps (ALLOCATE+RELEASE same node) |
| `gradients` | Activation grad wavefronts (transient) |
| `weight_grads` | PERSIST weight accum buffers |
| `state` | KV cache, optimizer moments, cross-step carried buffers |

---

## 6. `LoweredGraph` carries parameters

**Rationale:** `LoweredGraph` is the refined costing artifact. Requiring a parallel `Graph` for parameter bytes breaks self-containment.

### 6.1 New types — `src/zepto/analysis/lowered.py`

```python
from zepto.graph.ids import ParameterId

@dataclass(frozen=True, slots=True)
class LoweredParameter:
    id: str                           # "p{index}"
    parameter_id: ParameterId
    tensor: Tensor                    # dtype resolved via AccountingPolicy
    role: TensorRole                  # TensorRole.PARAMETER
    storage_id: str
    trainable: bool

@dataclass(frozen=True, slots=True)
class LoweredGraph:
    edges: Mapping[str, LoweredEdge]
    parameters: Mapping[str, LoweredParameter]           # NEW
    parameter_map: Mapping[ParameterId, str]             # NEW
    nodes: tuple[LoweredNode, ...]
    context: InvocationContext
    edge_map: Mapping[EdgeId, str]
    node_map: Mapping[NodeId, str]
    output_edge_ids: tuple[str, ...]                       # NEW — lowered output ids
    selections: tuple[ImplementationSelection, ...]
    fusion_map: Mapping[NodeId, str] = MappingProxyType({})
    region_map: Mapping[str, tuple[NodeId, ...]] = MappingProxyType({})
    region_selections: tuple[RegionImplementationSelection, ...] = ()
    state_port_events: tuple[StatePortEvent, ...] = ()     # Phase 3
```

### 6.2 Lowering registration — `src/zepto/analysis/lowering/transform.py`

Add after input edge registration in `lower()`:

```python
def _register_parameters(
    graph: Graph,
    context: InvocationContext,
    state: LoweringState,
) -> None:
    from ..lowered import LoweredParameter
    from zepto.graph.parameter import parameter_as_tensor

    for param_id, param in graph.parameters.items():
        lowered_id = f"p{param_id.index}"
        tensor, role = resolve_for_accounting(
            parameter_as_tensor(param),
            role_ctx=RoleContext(graph=graph, port_direction="parameter"),
            context=context,
        )
        storage_id = f"param:{param_id.index}"
        state.lowered_parameters[lowered_id] = LoweredParameter(
            id=lowered_id,
            parameter_id=param_id,
            tensor=tensor,
            role=role,
            storage_id=storage_id,
            trainable=param.trainable,
        )
        state.parameter_map[param_id] = lowered_id
```

Extend `LoweringState`:

```python
@dataclass
class LoweringState:
    lowered_edges: dict[str, LoweredEdge] = field(default_factory=dict)
    lowered_parameters: dict[str, LoweredParameter] = field(default_factory=dict)  # NEW
    parameter_map: dict[ParameterId, str] = field(default_factory=dict)              # NEW
    ...
```

Populate `output_edge_ids` from `graph.outputs` via `edge_map`.

### 6.3 API consequence

```python
# BEFORE (rejected):
account_memory(lowered, graph)

# AFTER:
account_memory(lowered)   # parameters from lowered.parameters
```

---

## 7. Package layout

```text
src/zepto/analysis/
├── reports/
│   ├── __init__.py
│   ├── attribution.py
│   ├── memory.py
│   ├── flops.py
│   ├── cost.py
│   └── horizon.py
├── memory/
│   ├── __init__.py
│   ├── simulator.py       # ResourceEventSimulator + Pass 2
│   ├── schedule.py        # last_use map, augment_timeline
│   └── account.py         # account_memory()
├── flops/
│   ├── __init__.py
│   └── account.py         # account_flops()
├── horizon/
│   ├── __init__.py
│   ├── spec.py
│   ├── state.py
│   ├── records.py
│   ├── simulate.py
│   └── account.py         # HorizonMemoryReducer, HorizonFlopReducer
├── optimizer.py           # OptimizerPolicy
├── estimation.py          # estimate(), estimate_horizon()
├── accounting.py          # optional MemoryAccountingPolicy
├── lowered.py             # LoweredParameter, extended LoweredGraph
└── __init__.py            # exports
```

---

## 8. Report types — `src/zepto/analysis/reports/`

### 8.1 `reports/attribution.py`

```python
@dataclass(frozen=True, slots=True)
class AttributionSlice:
    key: str
    forward_flops: int = 0
    backward_flops: int = 0
    allocated_bytes: int = 0
    peak_bytes: int = 0
    persistent_bytes: int = 0
    workspace_bytes: int = 0
```

### 8.2 `reports/memory.py`

```python
@dataclass(frozen=True, slots=True)
class MemoryBreakdown:
    activations: int = 0
    parameters: int = 0
    workspace: int = 0
    saved_for_backward: int = 0
    persistent_inputs: int = 0
    gradients: int = 0
    weight_grads: int = 0
    state: int = 0

@dataclass(frozen=True, slots=True)
class MemoryReport:
    sum_all_bytes: int
    peak_live_bytes: int
    breakdown: MemoryBreakdown
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
    live_timeline: tuple[tuple[int, int], ...] | None = None  # debug/tests
```

### 8.3 `reports/flops.py`

```python
@dataclass(frozen=True, slots=True)
class FlopReport:
    forward_flops: int
    backward_flops: int
    total_flops: int
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
```

Total selection by `context.phase`: forward-only / backward-only / both.

### 8.4 `reports/cost.py`

```python
@dataclass(frozen=True, slots=True)
class CostReport:
    memory: MemoryReport
    flops: FlopReport
    context: InvocationContext

@dataclass(frozen=True, slots=True)
class HorizonCostReport:
    peak_vram: int
    total_flops: int
    per_step: tuple[CostReport, ...]
    state_final: StateSnapshot
    memory: HorizonMemoryReport
    flops: HorizonFlopReport
```

### 8.5 `reports/horizon.py`

```python
@dataclass(frozen=True, slots=True)
class HorizonMemoryReport:
    peak_live_bytes: int          # timeline peak (NOT max(step.peak))
    sum_all_bytes: int
    persistent_carry_bytes: int
    per_step: tuple[MemoryReport, ...]
    breakdown: MemoryBreakdown

@dataclass(frozen=True, slots=True)
class HorizonFlopReport:
    total_forward_flops: int
    total_backward_flops: int
    total_flops: int
    per_step: tuple[FlopReport, ...]
```

---

## 9. Memory accounting — `src/zepto/analysis/memory/`

### 9.1 `memory/simulator.py` (sketch)

```python
class ResourceEventSimulator:
    def __init__(self, lowered: LoweredGraph) -> None:
        self._lowered = lowered
        self._context = lowered.context

    def run(self) -> SimulationResult:
        timeline: list[TimelineEvent] = []
        slots: dict[str, StorageSlot] = {}

        self._bootstrap_parameters(slots, timeline)
        self._bootstrap_inputs(slots, timeline)
        self._bootstrap_state_ports(slots, timeline)

        for node_index, node in enumerate(self._lowered.nodes):
            if not self._node_active(node):
                continue
            for event in node.resource_events:
                self._apply_event(event, node_index, node, slots, timeline)
            self._node_cleanup(node_index, node, slots, timeline)

        last_use = build_last_use_map(self._lowered)
        naive_peak = self._peak_from_timeline(timeline)
        augmented = augment_timeline_with_scheduled_releases(
            timeline, last_use, self._lowered
        )
        peak = self._peak_from_timeline(augmented)

        return SimulationResult(
            sum_all_bytes=self._sum_all,
            peak_live_bytes=peak,
            peak_live_bytes_naive=naive_peak,
            breakdown=self._breakdown,
            timeline=tuple(augmented),
            by_module=self._rollup_by_module(),
            by_region=self._rollup_by_region(),
            by_implementation=self._rollup_by_implementation(),
        )
```

### 9.2 `memory/account.py`

```python
def account_memory(
    target: LoweredGraph | HorizonSimulation,
) -> MemoryReport | HorizonMemoryReport:
    if isinstance(target, LoweredGraph):
        result = ResourceEventSimulator(target).run()
        return MemoryReport(
            sum_all_bytes=result.sum_all_bytes,
            peak_live_bytes=result.peak_live_bytes,
            breakdown=result.breakdown,
            by_module=result.by_module,
            by_region=result.by_region,
            by_implementation=result.by_implementation,
            live_timeline=result.live_timeline,
        )
    return account_horizon_memory(target)
```

---

## 10. FLOP accounting — `src/zepto/analysis/flops/account.py`

```python
def account_flops(
    target: LoweredGraph | HorizonSimulation,
) -> FlopReport | HorizonFlopReport:
    if isinstance(target, LoweredGraph):
        forward = sum(n.forward_flops for n in target.nodes)
        backward = sum(n.backward_flops for n in target.nodes)
        total = _select_total(forward, backward, target.context.phase)
        return FlopReport(
            forward_flops=forward,
            backward_flops=backward,
            total_flops=total,
            by_module=_rollup_flops_by_module(target),
            by_region=_rollup_flops_by_region(target),
            by_implementation=_rollup_flops_by_implementation(target),
        )
    return account_horizon_flops(target)


def _select_total(forward: int, backward: int, phase: str) -> int:
    if phase == "forward":
        return forward
    if phase == "backward":
        return backward
    return forward + backward
```

---

## 11. Horizon — `src/zepto/analysis/horizon/`

### 11.1 `horizon/spec.py`

```python
class StepKind(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    MICRO_FORWARD = "micro_forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"
    GENERIC = "generic"

@dataclass(frozen=True, slots=True)
class KVConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: DType

@dataclass(frozen=True, slots=True)
class HorizonStep:
    kind: StepKind
    seq_len: int
    batch: int = 1
    phase: str = "forward"
    name: str = ""
    attention_backend: str | None = None   # decode steps only
    decode_index: int | None = None

@dataclass
class HorizonSpec:
    steps: list[HorizonStep] = field(default_factory=list)
    kv: KVConfig | None = None
    optimizer_policy: OptimizerPolicy | None = None

    @classmethod
    def inference(cls, *, prefill: int, decode_steps: int = 0, kv: KVConfig | None = None, ...) -> HorizonSpec: ...

    @classmethod
    def training(cls, *, seq_len: int, micro_batches: int = 1, optimizer: OptimizerPolicy | None = AdamW) -> HorizonSpec: ...

    @classmethod
    def repeat(cls, invocations: int, *, seq_len: int, ...) -> HorizonSpec: ...

    def build_state_registry(self) -> StatePortRegistry: ...
    # fluent builders: prefill(), decode(), grad_accum(), backward_step(), training_step()
```

### 11.2 `horizon/state.py`

```python
@dataclass(frozen=True, slots=True)
class KVCacheState:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    dtype: DType

    @property
    def bytes(self) -> int: ...

@dataclass(frozen=True, slots=True)
class GradAccumState:
    parameter_bytes: int          # total grad buffer size
    micro_batches_seen: int

@dataclass(frozen=True, slots=True)
class OptimizerState:
    policy: OptimizerPolicy
    bytes: int

@dataclass(frozen=True, slots=True)
class StateSnapshot:
    kv_caches: tuple[KVCacheState, ...] = ()
    grad_accum: GradAccumState | None = None
    optimizer: OptimizerState | None = None
    custom: tuple[tuple[str, object], ...] = ()

class StatePortRegistry:
    def bind_to_context(
        self, base: InvocationContext
    ) -> InvocationContext: ...

    def advance(self, lowered: LoweredGraph) -> StatePortRegistry: ...
```

### 11.3 `horizon/simulate.py`

```python
ModuleFn = Module | Callable[[Compose], Module]
InputsFn = Callable[[HorizonStep, InvocationContext, StateSnapshot], tuple[Tensor, ...]]

def simulate_horizon(
    spec: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> HorizonSimulation:
    port_registry = spec.build_state_registry()
    timeline: list[InvocationRecord] = []
    initial = port_registry.snapshot()

    for step in spec.steps:
        ctx = _merge_context(step, context, port_registry)
        inputs = inputs_fn(step, ctx, port_registry.snapshot())
        graph = compose_graph(module_fn, inputs)
        lowered = lower(graph, ctx, registry=registry)
        state = state.advance(lowered)
        timeline.append(
            InvocationRecord(step=step, graph=graph, lowered=lowered)
        )

    return HorizonSimulation(
        spec=spec,
        timeline=tuple(timeline),
        state_initial=initial,
        state_final=state.snapshot(),
        base_context=base_context,
    )
```

**Degenerate case:** one-step spec must match single-invocation accounting exactly.

### 11.4 `horizon/account.py` — timeline reducer

```python
class HorizonMemoryReducer:
    """
    Merge per-step timelines with carried persistent storage.

    Rules:
    - Parameter bytes: counted once (from step 0 or dedicated bootstrap).
    - KV cache: monotonic growth; peak reflects largest cache seen.
    - Optimizer state: live after first optimizer step until horizon end.
    - Grad accum buffers: live across micro-batch forwards until backward consumes.
    - peak_live_bytes = max over merged timeline — NOT max(per_step.peak_live_bytes)
    """

    def reduce(
        self, sim: HorizonSimulation
    ) -> HorizonMemoryReport: ...
```

---

## 12. Optimizer — `src/zepto/analysis/optimizer.py`

```python
@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    name: str
    state_bytes_per_parameter: int    # e.g. 8 for AdamW (m + v in fp32)
    workspace_bytes: int = 0

AdamW = OptimizerPolicy(name="adamw", state_bytes_per_parameter=8)
SGD = OptimizerPolicy(name="sgd", state_bytes_per_parameter=0)
```

Horizon `optimizer` step allocates `PERSIST` optimizer state slots via `StatePortRegistry`.

---

## 13. Facade — `src/zepto/analysis/estimation.py`

```python
def estimate(
    graph: Graph,
    context: InvocationContext,
    *,
    return_lowered: bool = False,
) -> CostReport | tuple[CostReport, LoweredGraph]:
    lowered = lower(graph, context)
    mem = account_memory(lowered)
    flops = account_flops(lowered)
    report = CostReport(memory=mem, flops=flops, context=context)
    return (report, lowered) if return_lowered else report


def estimate_horizon(
    horizon: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    return_simulation: bool = False,
) -> HorizonCostReport | tuple[HorizonCostReport, HorizonSimulation]:
    sim = simulate_horizon(horizon, module_fn, inputs_fn, context)
    hmem = account_memory(sim)
    hflops = account_flops(sim)
    report = HorizonCostReport(
        peak_vram=hmem.peak_live_bytes,
        total_flops=hflops.total_flops,
        per_step=tuple(
            CostReport(
                memory=step_mem,
                flops=step_flop,
                context=rec.step.context(context, sim.state_initial),
            )
            for rec, step_mem, step_flop in zip(
                sim.timeline, hmem.per_step, hflops.per_step, strict=True
            )
        ),
        state_final=sim.state_final,
        memory=hmem,
        flops=hflops,
    )
    return (report, sim) if return_simulation else report
```

---

## 14. Lowering extensions (Phase 3)

### 14.1 `lower()` signature

```python
def lower(
    graph: Graph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
    state_ports: StatePortRegistry | None = None,
) -> LoweredGraph:
```

### 14.2 GQA KV state — `gqa/variants.py`

When `context.state` includes KV cache:

- **Prefill:** `ALLOCATE` + `PERSIST` for new KV tensors per layer.
- **Decode:** extend cache by one token; read full cached length for attention FLOPs.

---

## 15. Public exports

### `src/zepto/analysis/__init__.py`

Add:

```python
from .memory import account_memory
from .flops import account_flops
from .reports import (
    CostReport, MemoryReport, FlopReport,
    HorizonCostReport, HorizonMemoryReport, HorizonFlopReport,
)
from .horizon import HorizonSpec, KVConfig, simulate_horizon, inputs_from_shape, HorizonSimulation
from .estimation import estimate, estimate_horizon
from .optimizer import OptimizerPolicy, AdamW, SGD
from .lowered import LoweredParameter
```

Mirror in `src/zepto/__init__.py`.

---

## 16. Build phases and agent handoff

### Phase 1a — Simulator core + parameter lowering + event helper fix

**Goal:** Correct train-peak on small graphs (MatMul → ReLU → MatMul) before Apertus.

| Task | Files | Done when |
|------|-------|-----------|
| Add `LoweredParameter`, extend `LoweredGraph` | `lowered.py` | Types compile; tests import |
| Register parameters in `lower()` | `transform.py`, `helpers.py` | `lower()` populates `parameters` |
| Fix grad event helpers | `semantic/operations/helpers.py` + all call sites | No PERSIST on activation grads |
| Implement `StorageSlot`, Pass 1 simulator | `memory/simulator.py` | Bootstrap + node events |
| Implement `build_last_use_map`, Pass 2 | `memory/schedule.py` | Scheduled RELEASE |
| Tests | `tests/test_memory_simulator.py` | See §17.1 |

**Agent entry point:** start at §5 and §6; run `pytest tests/test_memory_simulator.py`.

**Do not start:** reports, horizon, GQA state, Apertus integration.

---

### Phase 1b — Reports + account APIs + estimate facade

| Task | Files |
|------|-------|
| Report dataclasses | `reports/*.py` |
| `account_memory`, `account_flops` | `memory/account.py`, `flops/account.py` |
| `estimate()` | `estimation.py` |
| Tests | `tests/test_memory_accounting.py`, `tests/test_flop_accounting.py`, `tests/test_estimation.py` |

**Depends on:** Phase 1a complete.

---

### Phase 2 — Horizon (inference + training)

| Task | Files |
|------|-------|
| `HorizonSpec`, steps | `horizon/spec.py` |
| State ports: KV, grad accum, optimizer | `horizon/state.py`, `optimizer.py` |
| `simulate_horizon` | `horizon/simulate.py` |
| Timeline reducer | `horizon/account.py`, `reports/horizon.py` |
| `estimate_horizon` | `estimation.py` |
| Tests | `tests/horizon/*` |

**Depends on:** Phase 1b complete.

**Training scope:** grad accumulation and optimizer state are required in this phase, not optional.

---

### Phase 3 — KV state in GQA + Apertus integration

| Task | Files |
|------|-------|
| State-aware GQA lowering | `gqa/variants.py`, `lowering/context.py`, `helpers.py` |
| `state_ports` on `lower()` | `transform.py` |
| Integration tests | `tests/integration/test_apertus_*` |

**Depends on:** Phase 2 complete.

---

## 17. Test plans

### 17.1 Phase 1a — `tests/test_memory_simulator.py`

| Test | Validates |
|------|-----------|
| `test_identity_alias_no_peak_increase` | ALIAS semantics |
| `test_forward_allocate_peak` | Basic ALLOCATE + peak |
| `test_save_pins_operand_through_forward_cleanup` | SAVE lifetime |
| `test_unreduced_temp_same_node_release` | Workspace within node |
| `test_parameter_bootstrap_in_peak` | LoweredParameter bytes live from t=0 |
| `test_persistent_input_bootstrap` | Causal mask PERSIST |
| `test_phase_forward_skips_backward_events` | Phase filter |
| `test_activation_grad_no_persist` | Grad port ALLOCATE only in events |
| `test_scheduled_release_lowers_peak` | Pass 2: peak < naive peak on chain |
| `test_matmul_relu_matmul_train_peak` | **Golden:** peak ≈ weights + one wavefront, not ~3× activations |

#### Golden test sketch: MatMul → ReLU → MatMul

```python
def test_matmul_relu_matmul_train_peak_not_sum_all():
    """
    Chain: X -> MatMul0 -> Y0 -> ReLU -> Y1 -> MatMul1 -> Y2
    All operands require grad. Backward pass requested.

    With PERSIST on every grad port (old behavior), peak ~ 3×|activation|.
    With schedule-driven release, peak ~ |weights| + |max wavefront|.
    """
    graph = build_matmul_relu_matmul_graph(
        batch=32, seq=128, hidden=512, inner=256
    )
    ctx = reference_invocation(phase="backward")  # or horizon split
    lowered = lower(graph, ctx)

    result = ResourceEventSimulator(lowered).run()

    weight_bytes = sum(
        lowered.context.accounting.bytes_for(
            ResolvedValue(p.tensor, p.role, p.tensor.dtype)
        )
        for p in lowered.parameters.values()
    )
    largest_activation = _max_activation_bytes(lowered)

    # Peak should be closer to weights + one activation + one grad wavefront
    assert result.peak_live_bytes < result.peak_live_bytes_naive
    assert result.peak_live_bytes <= weight_bytes + 3 * largest_activation
    # Tight bound once golden value is recorded:
    # assert result.peak_live_bytes == pytest.approx(expected_peak, rel=0.01)
```

### 17.2 Phase 1b tests

- Reuse fixtures from `tests/test_lowering.py`
- `estimate(graph, ctx)` matches explicit `lower` + `account_*` chain
- FLOP totals match sum of `node.forward_flops` / `node.backward_flops`

### 17.3 Phase 2 horizon tests

| Test | Validates |
|------|-----------|
| `test_one_step_degenerates_to_single_invocation` | Horizon with 1 step == direct path |
| `test_prefill_decode_peak_includes_kv` | KV grows; timeline peak > any single step |
| `test_horizon_peak_not_max_step_peak` | Carried weights counted once |
| `test_grad_accum_carries_buffers` | Micro-batch forwards retain grad accum |
| `test_optimizer_state_persisted` | Optimizer step adds PERSIST state |

### 17.4 Phase 3 integration

- `tests/integration/test_apertus_prefill_estimate.py`
- `tests/integration/test_apertus_prefill_decode_horizon.py`

---

## 18. Locked design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `LoweredGraph` carries parameters | Self-contained costing artifact |
| 2 | `account_memory(lowered)` — no `Graph` arg | Parameters on lowered graph |
| 3 | Pass 2 schedule augmentation required for peak | Atto alignment; fixes ~3× FFN over-count |
| 4 | Activation grad: ALLOCATE only in events; RELEASE in simulator | Producer-dependent lifetime |
| 5 | Weight grad accum: ALLOCATE + PERSIST | Cross-node / horizon persistence |
| 6 | Horizon peak = merged timeline peak | NOT `max(step.peak)` |
| 7 | One graph per horizon step | ADR-0003, ADR-0005 |
| 8 | Training (grad accum, optimizer) in Phase 2 | User requirement |
| 9 | Separate `estimate_horizon()` name | Transparency over overloaded `estimate()` |
| 10 | Phase 1a scope: small-graph train peak before Apertus | De-risk simulator correctness |

---

## 19. Open questions (resolve during Phase 1a)

1. **`phase="full"` lowering:** emit both event sets in one `lower()` call, or model training as horizon `[forward_step, backward_step]`? Recommendation: horizon split first; `phase="full"` later.

2. **`last_use` for fused regions:** region nodes subsume multiple structural nodes — use region boundary edges and `saved_for_backward` on fused `LoweredNode`.

3. **Graph outputs at invocation end:** remain live (count toward peak at end) but excluded from node cleanup — map via `lowered.output_edge_ids`.

---

## 20. Reference: current code touchpoints

| Area | Path | Notes |
|------|------|-------|
| Event kinds | `src/zepto/semantic/operations/records.py` | `ResourceEventKind` |
| Grad helpers (to fix) | `src/zepto/semantic/operations/helpers.py` | `persist_only_events`, `backward_gradient_port_events` |
| SAVE injection | `src/zepto/analysis/lowering/helpers.py` | `append_save_events` |
| Lowered records | `src/zepto/analysis/lowered.py` | extend here |
| Lowering entry | `src/zepto/analysis/lowering/transform.py` | parameter registration |
| Stub facade | `src/zepto/analysis/estimation.py` | replace stub |
| ADR memory | `docs/adr/0006-tensor-accounting-and-memory-reports.md` | sum-all vs peak |
| ADR horizon | `docs/adr/0003-resource-events-and-horizon-simulation.md` | one graph per invocation |
| Domain terms | `CONTEXT.md` | Memory report, state port, horizon |

---

## 21. Suggested ADR follow-up

After Phase 1a lands, add `docs/adr/0010-estimation-api-and-horizon.md` formalizing:

- Pass 1 / Pass 2 simulator split
- Activation grad vs weight grad event conventions
- Horizon peak reduction semantics
- `LoweredGraph.parameters` invariant

---

*End of implementation plan.*
