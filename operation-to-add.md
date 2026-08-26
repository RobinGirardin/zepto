# Primitive operations to add

This document lists **atomic** Zepto `Operation` declarations still missing from the
current primitive set, and explains why each belongs in the operation layer rather
than as a `Module`, as reuse of an existing similar primitive, or as a lowered
implementation of another operation.

## Design rule

A Zepto **Operation** must be **irreducible**: its forward semantics, backward
contract, and estimation behavior cannot be expressed as a composition of other
structural operations without changing meaning, cost, or saved-state rules.

Consequences:

| Layer | Responsibility |
|-------|----------------|
| **Operation** | Atomic semantic + estimation unit; one graph node per call |
| **Module** | Composes operations; provenance boundary; may fuse at lowering |
| **Implementation** | Backend-specific execution of one op or a matched **region** of ops |

Atto exposes several **named IR nodes** (e.g. `RMSNorm`, `Softmax`, `XIELU`) that
are **recipe-level or fused-cost leaves**, not mathematical atoms. Zepto should
match Atto **estimation outcomes** via Modules plus lowering, not by copying every
Atto op name into `core/operation/`.

`PLAN.md` §6 already commits `LayerNorm`, `RMSNorm`, and `RoPE` to
`zepto.modules` as primitive compositions. This document extends that rule to the
full Atto recipe surface.

## Current Zepto primitives (baseline)

Already shipped in `src/zepto/core/operation/`:

`Add`, `Subtract`, `Multiply`, `Divide`, `MatMul`, `Maximum`, `Minimum`,
`SquareRoot`, `Identity`, `Reshape`, `Transpose`, `Split`

These cover elementwise arithmetic, GEMM, unary sqrt, and zero-FLOP layout ops
with **fixed numel** (reshape / transpose / split).

---

## Operations to add (11)

Each entry states why the op is atomic, why the rejected alternatives fail, and
includes the planned **operational contract** (full `Operation` declaration per
[ADR-0004](docs/adr/0004-complete-operation-contracts.md)).

**How to read each operation.** Every op below has a **How it works** section
with four parts, written so you do not need to already know the op:

- **Forward** — what the output is
- **Backward** — how the output's gradient becomes each input's gradient
- **Broadcast** — whether differently shaped inputs are allowed, and what that
  does
- **Memory** — which forward values to keep, and which extra gradient buffers
  to allocate

When the op is implemented, copy that section into
[docs/primitive-backward.md](docs/primitive-backward.md) as a new heading and
keep the two texts the same.

Shared rules (same as the existing primitives):

- An input's gradient always has the **same shape** as that input.
- If one value was reused at many output positions, its gradient is the
  **sum** of those positions, never the average.
- A full-size temporary (output-shaped, then thrown away) is only needed when
  backward first **computes a new tensor the size of the output**, then sums it
  down — the same pattern as `Multiply`. If backward only copies, stretches,
  slices, or sums the incoming gradient, there is no such temporary — the same
  pattern as `Add`.

How shapes behave, in short:

| What changes | Ops | Forward | Backward |
|--------------|-----|---------|----------|
| Nothing (same shape) | `Exp`, `Log`, `Sin`, `Cos` | One formula per element | The matching derivative per element |
| Stack or look up along an axis | `Concat`, `Gather` | Join pieces, or pick rows by index | Split apart, or put each piece back |
| Two inputs stretched to one shape | `Pow` | Stretch, then raise to a power | Compute at full size, then add extras back |
| Three inputs stretched to one shape | `Where` | Stretch, then pick true or false | Mask at full size, then add extras back |
| Collapse an axis | `ReduceSum` | Add along that axis | Copy the short gradient back out |
| Repeat an axis | `RepeatKV` | Copy that slice several times | Add the copies back together |

### 1. `ReduceSum`

**Family:** `reduce_sum`

**Semantics:** Add up all values along one or more chosen axes. Optionally keep
those axes as size 1 so the output still has the same number of dimensions.

**Why a dedicated Operation**

- Reduction along an axis is not expressible from elementwise binaries (`Add`
  only combines two tensors, not an arbitrary axis fold).
- Required for normalization statistics, softmax denominators, and broadcast
  gradient reductions referenced in `docs/primitive-backward.md`.

**Why not a Module**

- Any “Module” that implements sum-over-axis would internally need this op; the
  Module would be a thin wrapper with no remaining decomposition target.

**Why not an existing primitive**

- `Add` does not generalize to n-ary axis reduction without a tree of pairwise
  adds (wrong FLOP count, wrong associativity for float, no single backward
  contract).

**Why not lowering only**

- Lowering fuses *execution*; the structural graph still needs one semantic
  reduction node so provenance, validation, and unfused estimation remain
  correct.

**Unlocks (as Modules):** `Softmax`, `LayerNorm`, `RMSNorm`, parts of
`CrossEntropy`, broadcast backward sum-reductions.

**How it works** (copy into `docs/primitive-backward.md` as `## ReduceSum`):

**Forward.** Add every entry along the chosen axes. Example: a `(3, 4)` table
summed along rows becomes a length-4 vector (one total per column). If we
keep the empty axis (`keepdim=True`) that result stays `(1, 4)` instead of
`(4,)`. Axis `-1` means the last axis. Listing the same axis twice is ignored. Summing no axes leaves
the tensor unchanged. A single number with no axes is rejected. Summing every
axis yields one number (shape `()`).

$$
Y = \mathrm{sum}(A, \mathrm{axes})
$$

**Backward.** Each input position went into exactly one output sum, so the
input's gradient is the output's gradient **copied** to every position that
was added together. Stretching a shorter tensor is not arithmetic, so backward
costs 0 FLOPs. Forward costs *(input elements − output elements)* additions
(adding *n* numbers takes *n*−1 adds).

$$
\frac{\partial L}{\partial A}
= \mathrm{copy}\!\left(\frac{\partial L}{\partial Y},\ \text{shape of } A\right)
$$

**Broadcast.** This op does not stretch two tensors together. It only sums
inside one tensor.

**Memory.** Keep nothing from the forward pass (which axes to sum is stored on
the op). Store the input's gradient at the input's shape. Do not allocate a
second buffer at the output's shape: the incoming gradient already has that
shape and is only stretched.

**Operational contract** (`src/zepto/core/operation/reduce_sum.py`):

```python
"""Axis reduction by sum."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    numel,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


def _normalize_axes(axis: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else axis
    normalized: list[int] = []
    for dim in axes:
        resolved = dim if dim >= 0 else rank + dim
        if resolved < 0 or resolved >= rank:
            raise ValueError(f"reduce_sum axis {dim} out of range for rank {rank}")
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def _reduce_output_shape(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    *,
    keepdim: bool,
) -> tuple[int, ...]:
    axes = set(_normalize_axes(axis, len(shape)))
    if keepdim:
        return tuple(1 if index in axes else dim for index, dim in enumerate(shape))
    return tuple(dim for index, dim in enumerate(shape) if index not in axes)


@dataclass(frozen=True, slots=True)
class ReduceSum(Operation):
    """Sum tensor elements along one or more axes."""

    axis: int | tuple[int, ...]
    keepdim: bool = False

    @property
    def family(self) -> str:
        return "reduce_sum"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (input_meta,) = inputs
        if not input_meta.shape:
            raise ValueError("reduce_sum requires a ranked input tensor")
        output_shape = _reduce_output_shape(
            input_meta.shape, self.axis, keepdim=self.keepdim
        )
        return (
            ValueMetadata(
                output_shape,
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        input_meta = context.metadata_for("input")
        output = context.metadata_for("output")
        if input_meta is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        return numel(input_meta) - numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 2. `Pow`

**Family:** `pow`

**Semantics:** Raise each value to a power: `base ** exponent`. The two inputs
may have different shapes; they are stretched to a common shape first, the same
way `Add` and `Multiply` do. The exponent can be one number or a whole tensor.

**Why a dedicated Operation**

- Distinct backward from chained `Multiply` (`∂/∂x` involves `exponent * x**(e-1)`
  and a separate exponent path).
- Needed for variance (`x**2`) in norms and for xIELU’s cubic term.

**Why not a Module**

- A Module of multiplies only covers integer exponents and misstates FLOPs and
  saved tensors for general powers.

**Why not `Multiply` / `SquareRoot`**

- `SquareRoot` is `x**0.5` but does not cover `x**2`, scalar exponents, or
  general exponent gradients.

**Why not lowering**

- Fused norm kernels may lower a region to one implementation, but the
  structural graph must still record `Pow` if unfused estimation is required.

**Unlocks:** variance in norm Modules, xIELU Module negative branch.

**How it works** (copy into `docs/primitive-backward.md` as `## Pow`):

**Forward.** For every position, raise the base to the exponent. If one input
is smaller, it is stretched by repeating (for example one exponent applied to
a whole tensor). Cost: one power per output element.

$$
Y = A^{B}
$$

**Backward.** Two separate formulas:

- Gradient for the base: incoming gradient × exponent × base^(exponent − 1).
  Needs the **base** and the **exponent**, not the output.
- Gradient for the exponent: incoming gradient × output × log(base). Needs
  the **base** and the **output**.

The log needs a positive base. A power with a negative or fractional exponent
is undefined at zero (same kind of caveat as square root).

Each formula costs 3 operations per output element, plus extra additions if
that input was stretched and must be summed back to its original shape.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * B * A^{B-1}
\qquad
\frac{\partial L}{\partial B}
= \frac{\partial L}{\partial Y} * Y * \log A
$$

**Broadcast.** Same stretching rules as `Multiply`. If an input was stretched
in the forward pass, backward **adds** those extra copies together so the
gradient matches the original input shape.

**Memory.** Like `Multiply`: if an input was stretched, first compute a
full-size gradient, then sum it down. That full-size temporary exists only for
a stretched input. Save `base` and `exponent` if the base needs a gradient.
Save `base` and `output` if the exponent needs a gradient.

**Operational contract** (`src/zepto/core/operation/pow.py`):

```python
"""Elementwise power operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_binary_auxiliary_ports,
    allocate,
    broadcast_metadata,
    broadcast_reduction_flops,
    backward_gradient_port_events,
    numel,
    persist_only_events,
    reduced_gradient_metadata,
    unreduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_BASE = "grad_base"
GRAD_EXPONENT = "grad_exponent"
GRAD_BASE_UNREDUCED = "grad_base_unreduced"
GRAD_EXPONENT_UNREDUCED = "grad_exponent_unreduced"


class Pow(Operation):
    """Raise base to exponent with NumPy-style broadcast semantics."""

    @property
    def family(self) -> str:
        return "pow"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("base"), PortSpec("exponent"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(GRAD_BASE, ValueKind.GRADIENT),
            PortSpec(GRAD_EXPONENT, ValueKind.GRADIENT),
            PortSpec(GRAD_BASE_UNREDUCED, ValueKind.GRADIENT),
            PortSpec(GRAD_EXPONENT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        base, exponent = inputs
        (output,) = outputs
        return (
            reduced_gradient_metadata(base),
            reduced_gradient_metadata(exponent),
            unreduced_gradient_metadata(output),
            unreduced_gradient_metadata(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        base, exponent = inputs
        (output,) = outputs
        rename = {
            GRAD_LEFT: GRAD_BASE,
            GRAD_RIGHT: GRAD_EXPONENT,
            GRAD_LEFT_UNREDUCED: GRAD_BASE_UNREDUCED,
            GRAD_RIGHT_UNREDUCED: GRAD_EXPONENT_UNREDUCED,
        }
        return tuple(
            rename[name]
            for name in active_binary_auxiliary_ports(
                base, exponent, output, materializes_vjp=True
            )
        )

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("base", "exponent", "output"),
            gradient_inputs=("output",),
            gradient_outputs=("base", "exponent"),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        base, exponent = inputs
        output = broadcast_metadata(
            (base, exponent),
            family=self.family,
            semantic_type="tensor",
        )
        return (output,)

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        base, exponent = inputs
        names: list[str] = []
        if base.requires_grad:
            names.extend(("base", "exponent"))
        if exponent.requires_grad:
            names.extend(("base", "output"))
        return tuple(dict.fromkeys(names))

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        base = context.metadata_for("base")
        exponent = context.metadata_for("exponent")
        output = context.metadata_for("output")
        if base is None or exponent is None or output is None:
            raise ValueError(
                "Estimation context must provide 'base', 'exponent', and 'output' ports"
            )
        flops = 0
        if base.requires_grad:
            flops += 3 * numel(output)
            if base.shape != output.shape:
                flops += broadcast_reduction_flops(base, output)
        if exponent.requires_grad:
            flops += 3 * numel(output)
            if exponent.shape != output.shape:
                flops += broadcast_reduction_flops(exponent, output)
        return flops

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        unreduced_by_port = {
            GRAD_BASE: GRAD_BASE_UNREDUCED,
            GRAD_EXPONENT: GRAD_EXPONENT_UNREDUCED,
        }
        active = set(result.active_auxiliary_ports)
        for port_name in (GRAD_BASE, GRAD_EXPONENT):
            if port_name not in active:
                continue
            unreduced = unreduced_by_port[port_name]
            if unreduced in active:
                events.extend(
                    backward_gradient_port_events(
                        port_name, unreduced_port=unreduced
                    )
                )
            else:
                events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 3. `Exp`

**Family:** `exp`

**Semantics:** Replace every value \(x\) with \(e^{x}\). The output has the same
shape as the input.

**Why a dedicated Operation**

- Unary transcendental with its own backward (`grad = grad_out * exp(x)` or
  reuse output).
- Cannot be derived from `Add` / `Multiply` / `Pow` over rationals.

**Why not a Module**

- Any Module named “Exp” would be a single-op passthrough — the op *is* the
  Module.

**Why not lowering of `Pow` with e**

- Base-e exp is the standard primitive; routing through `Pow(base=e, x)` hides
  the estimation leaf Atto uses (0 FLOP trig/exp conventions in some paths) and
  complicates `saved_for_backward`.

**Unlocks:** `Softmax` Module, xIELU Module, RoPE materialize Module (Atto bills
trig at 0 FLOP but still needs semantic nodes).

**How it works** (copy into `docs/primitive-backward.md` as `## Exp`):

**Forward.** Apply \(e^{x}\) to every element. Same shape as the input. We bill
**0 FLOPs**: Zepto's cost model follows PyTorch's `FlopCounterMode` convention,
which does not count transcendental ops (`exp`, `log`, `sin`, `cos`) — same
treatment as `Sin`/`Cos`.

$$
Y = \exp(A)
$$

**Backward.** Multiply the incoming gradient by the **output** (which is already
\(e^{x}\)), instead of computing exp again. We still bill **0 FLOPs** for both
forward and backward (same free-transcendental convention as `Sin`/`Cos`). The
formula itself does not change — the output is still worth saving, to avoid
recomputing `exp`.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * Y
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the output if the input needs a gradient. Store the input's
gradient at the input's shape. No extra temporary.

**Operational contract** (`src/zepto/core/operation/exp.py`):

```python
"""Elementwise natural exponential operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


class Exp(Operation):
    """Apply an elementwise natural exponential."""

    @property
    def family(self) -> str:
        return "exp"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("output",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("output",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 4. `Log`

**Family:** `log`

**Semantics:** Replace every value \(x\) with \(\log x\) (natural log). The
output has the same shape as the input.

**Why a dedicated Operation**

- Pair to `Exp` for log-space stabilization; required for cross-entropy and
  log-softmax compositions.
- Backward `grad/x` is not a composition of existing ops without division by
  `x` (division exists, but the **log** forward map is still missing).

**Why not a Module**

- Same passthrough argument as `Exp`.

**Why not `Pow` / algebraic rewrite**

- No inverse primitive exists today; log is not reconstructible from shipped
  ops.

**Unlocks:** `CrossEntropy` Module, numerically stable `Softmax` Module variants.

**How it works** (copy into `docs/primitive-backward.md` as `## Log`):

**Forward.** Apply \(\log x\) to every element. Same shape as the input. The
input should be positive. We bill **0 FLOPs** — same free-transcendental
convention as `Exp`/`Sin`/`Cos` (PyTorch's `FlopCounterMode` doesn't count
these either).

$$
Y = \log(A)
$$

**Backward.** Divide the incoming gradient by the **input** \(x\) (not by the
output). We still bill **0 FLOPs** for both forward and backward (same
convention as `Exp`).

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \frac{1}{A}
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

**Operational contract** (`src/zepto/core/operation/log.py`):

```python
"""Elementwise natural logarithm operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


class Log(Operation):
    """Apply an elementwise natural logarithm."""

    @property
    def family(self) -> str:
        return "log"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("input",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 5. `Sin`

**Family:** `sin`

**Semantics:** Replace every value \(x\) with \(\sin x\). The output has the
same shape as the input.

**Why a dedicated Operation**

- Trigonometric leaf; Atto RoPE materialize and apply decompose to sin/cos
  (billed at 0 arithmetic FLOP but distinct graph semantics).

**Why not a Module**

- Module would wrap one irreducible unary.

**Why not lowering of something else**

- Cannot express sin from algebraic ops already in Zepto.

**Unlocks:** `RoPEMaterialize` Module, `RoPE` apply Module (with `Cos`).

**How it works** (copy into `docs/primitive-backward.md` as `## Sin`):

**Forward.** Apply sine to every element. Same shape as the input.

$$
Y = \sin(A)
$$

**Backward.** Multiply the incoming gradient by \(\cos(x)\). That needs the
**input**, not the output (you cannot recover the sign of cosine from sine
alone). We still bill **0 FLOPs** for both forward and backward (trig is
treated as free in this cost model). The formula itself does not change.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \cos(A)
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

**Operational contract** (`src/zepto/core/operation/sin.py`):

```python
"""Elementwise sine operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


class Sin(Operation):
    """Apply an elementwise sine."""

    @property
    def family(self) -> str:
        return "sin"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("input",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 6. `Cos`

**Family:** `cos`

**Semantics:** Replace every value \(x\) with \(\cos x\). The output has the
same shape as the input.

**Why a dedicated Operation**

- Same reasoning as `Sin`; RoPE rotation uses both.

**Why not a Module / reuse `Sin`**

- Phase shift `sin(x + π/2)` would require π constants, adds, and pollutes
  estimation; cos has identical cost class but distinct backward (`-sin`).

**Unlocks:** RoPE Modules.

**How it works** (copy into `docs/primitive-backward.md` as `## Cos`):

**Forward.** Apply cosine to every element. Same shape as the input.

$$
Y = \cos(A)
$$

**Backward.** Multiply the incoming gradient by \(-\sin(x)\). That needs the
**input**. Forward and backward are billed at **0 FLOPs** (same free-trig
convention as `Sin`).

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \bigl(-\sin(A)\bigr)
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

**Operational contract** (`src/zepto/core/operation/cos.py`):

```python
"""Elementwise cosine operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


class Cos(Operation):
    """Apply an elementwise cosine."""

    @property
    def family(self) -> str:
        return "cos"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("input",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 7. `Concat`

**Family:** `concat`

**Semantics:** Place two or more tensors end-to-end along one chosen axis. All
other axes must already match; they are not stretched.

**Why a dedicated Operation**

- Structural inverse of `Split`; `docs/primitive-backward.md` defines split
  backward as concat of chunk gradients.
- Without `Concat`, `Split`’s backward contract cannot be structurally represented.

**Why not a Module**

- A concat Module would emit exactly one `Concat` op — no further decomposition.

**Why not `Reshape` / `Transpose`**

- Those preserve total numel; concat increases numel along an axis.

**Why not lowering-only**

- Split backward needs an explicit concat node in the structural graph for
  gradient routing and validation.

**Unlocks:** split backward, optional head-merge patterns; Atto `Concat` parity.

**How it works** (copy into `docs/primitive-backward.md` as `## Concat`):

**Forward.** Stack at least two tensors along one axis, like placing strips
side by side. They must have the same number of dimensions, and every axis
*except* the join axis must have the same size. Along the join axis, the
output length is the sum of the input lengths. A single number with no axes is
rejected. No arithmetic: 0 FLOPs.

$$
Y = \mathrm{concat}(A_0,\ldots,A_{n-1},\ \mathrm{axis})
\quad (n \ge 2)
$$

**Backward.** Cut the output's gradient back into the same-sized pieces and
hand each piece to the matching input. No arithmetic: 0 FLOPs. Today's `Split`
only cuts axis 0; concat on another axis is still valid, but is not the reverse
of that `Split` until `Split` also takes an axis.

$$
\frac{\partial L}{\partial A_k}
= \text{the }k\text{-th slice of }\frac{\partial L}{\partial Y}
\text{ along the join axis}
$$

**Broadcast.** None. Other axes must match exactly; a size-1 axis is not
stretched to match a larger one.

**Memory.** Keep nothing (the piece sizes are already known from the input
shapes). Store one gradient per input that needs one, at that input's shape.
Inputs that do not need a gradient still count toward the cut positions, so
the remaining pieces line up.

**Operational contract** (`src/zepto/core/operation/concat.py`):

```python
"""Multi-input concatenation operation declaration."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, persist_only_events, reduced_gradient_metadata
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"concat axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class Concat(Operation):
    """Concatenate two or more tensors along one axis."""

    axis: int
    input_count: int

    @property
    def family(self) -> str:
        return "concat"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return tuple(
            PortSpec(f"input_{index}") for index in range(self.input_count)
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return tuple(
            PortSpec(f"grad_input_{index}", ValueKind.GRADIENT)
            for index in range(self.input_count)
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return tuple(reduced_gradient_metadata(value) for value in inputs)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return tuple(
            f"grad_input_{index}"
            for index, value in enumerate(inputs)
            if value.requires_grad
        )

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=tuple(
                f"input_{index}" for index in range(self.input_count)
            ),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.input_count < 2:
            raise ValueError("concat requires at least two input tensors")
        if len(inputs) != self.input_count:
            raise ValueError(
                f"concat expects {self.input_count} inputs, got {len(inputs)}"
            )
        rank = len(inputs[0].shape)
        if rank == 0:
            raise ValueError("concat requires ranked input tensors")
        axis = _normalize_axis(self.axis, rank)
        output_shape = list(inputs[0].shape)
        output_shape[axis] = 0
        for index, value in enumerate(inputs):
            if len(value.shape) != rank:
                raise ValueError("concat inputs must share rank")
            for dim_index, (left, right) in enumerate(
                zip(inputs[0].shape, value.shape, strict=True)
            ):
                if dim_index == axis:
                    output_shape[axis] += right
                elif left != right:
                    raise ValueError(
                        f"concat non-concat dimensions mismatch on input_{index}: "
                        f"{inputs[0].shape} vs {value.shape}"
                    )
        requires_grad = any(value.requires_grad for value in inputs)
        return (
            ValueMetadata(
                tuple(output_shape),
                inputs[0].semantic_type,
                requires_grad=requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 8. `Gather`

**Family:** `gather`

**Semantics:** Look up slices of a tensor by integer indices along one axis —
like picking rows out of a table. Used for embedding lookup: a vocabulary
table of shape `(vocab, d)` plus token ids of shape `(batch, seq)` becomes
`(batch, seq, d)`.

**Why a dedicated Operation**

- **Embedding lookup** is gather on \(W_E \in \mathbb{R}^{V \times d}\) with
  token indices — not a `MatMul` (Atto: 0 FLOP gather; backward scatter-add
  into weight rows).
- Memory access pattern and port kinds (metadata index tensor vs activation
  weight) differ from GEMM.

**Why not a Module only**

- `EmbeddingLookup` **Module** is the right user API, but it must call one
  `Gather` operation internally; gather itself is not decomposable into matmul
  without sparsity explosion and wrong FLOPs.

**Why not `MatMul` one-hot**

- One-hot matmul materializes `(S, V)` — wrong VRAM and FLOP model.

**Why not lowering of MatMul**

- Lowering cannot change semantic family from matmul to gather; estimation rules
  diverge completely.

**Unlocks:** `Embedding` Module, label indexing inside `CrossEntropy` Module.

**How it works** (copy into `docs/primitive-backward.md` as `## Gather`):

**Forward.** Along a chosen axis, replace that axis with the index tensor's
shape, and at each output position copy the input slice whose number is in
`index`. Example: input rows `(5, 8)`, indices `[0, 4, 0]` along axis 0 →
output shape `(3, 8)` (row 0, row 4, row 0 again). Indices are integers in
range. This is ordinary “pick these rows,” not the other gather that requires
the index to have as many dimensions as the table. A table stored as
`(d, vocab)` would need a different axis plus a later transpose; that is the
Module's job, not this op. Cost: 0 FLOPs.

**Backward.** Send each output gradient back to the row it was copied from. If
the same row was picked more than once, **add** those gradients together. The
index list has no gradient (they are discrete choices). The input values
themselves are not needed — only the indices. Cost: 0 FLOPs (collision adds
are not billed).

$$
\frac{\partial L}{\partial A}
= \text{put each } \tfrac{\partial L}{\partial Y} \text{ back at } I
\text{ and add duplicates}
$$

**Broadcast.** The index is not stretched against the input. It *replaces* the
chosen axis: output shape is “input, with that axis swapped for the index
shape.”

**Memory.** Keep `index` if the input needs a gradient. Store the input's
gradient at the input's shape (start from zeros, then add each piece back).
Do not treat this as a view of the output's gradient: even when the two
tensors have the same number of elements, repeats like `[0, 0, 0]` still add
into one row. No extra full-size temporary.

**Operational contract** (`src/zepto/core/operation/gather.py`):

```python
"""NumPy-take / index-select operation declaration."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, persist_only_events, reduced_gradient_metadata
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"gather axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class Gather(Operation):
    """Select slices from input along axis (NumPy take / embedding lookup)."""

    axis: int

    @property
    def family(self) -> str:
        return "gather"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"), PortSpec("index"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        input_meta, _index = inputs
        return (reduced_gradient_metadata(input_meta),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("index",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        input_meta, index_meta = inputs
        if not input_meta.shape:
            raise ValueError("gather requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_meta.shape))
        output_shape = list(input_meta.shape)
        output_shape[axis : axis + 1] = list(index_meta.shape)
        return (
            ValueMetadata(
                tuple(output_shape),
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("index",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 9. `RepeatKV`

**Family:** `repeat_kv`

**Semantics:** Copy one axis several times. Default `axis=0` turns
`(n_kv, seq, head_dim)` into `(n_kv * n_rep, seq, head_dim)`. For grouped-query
attention on a batched tensor, use `axis=1`:
`(batch, n_kv, seq, head_dim)` → `(batch, n_kv * n_rep, seq, head_dim)`.
`n_rep = 1` is a no-op (the output is the same storage as the input).
Each original slice is copied `n_rep` times in a row.

**Why a dedicated Operation**

- **Numel-changing** layout distinct from `Reshape` (same numel) and
  `Transpose` (permutation).
- GQA (Apertus) relies on this exact pattern for head replication before
  attention matmuls; Atto assigns 0 FLOP but non-trivial alias/grad-reduce
  semantics.

**Why not a Module of `Concat`**

- Repeating the same tensor `n_rep` times along an axis can be modeled as
  `Concat` of the same tensor handle bound to `n_rep` input ports, and
  `Concat`'s split-backward would hand each port its own slice of the
  incoming gradient. But summing those `n_rep` slices back into **one**
  gradient for the shared input requires a graph-level rule that sums
  contributions across every consumer of a reused tensor. Zepto has not
  implemented (or documented as planned) any such cross-operation
  accumulation mechanism today — `Tensor.consumers` records that a tensor
  can fan out to multiple ports structurally, but nothing walks those edges
  and sums partial gradients during backward. Until that machinery exists,
  this decomposition would silently drop `n_rep - 1` of the gradient's
  contributions, so it is not meaning-preserving under Zepto's current
  primitives. Revisit this reclassification once a general multi-consumer
  gradient-accumulation pass lands.

**Why not `Reshape` + `Broadcast` multiply**

- Broadcast multiply changes values unless using ones-tensor tricks; still wrong
  for “logical copy along head axis” cost model.

**Why not lowering of Reshape**

- Structural op family must exist for GQA provenance and for unfused peak
  accounting; fusion may merge repeats later.

**Unlocks:** GQA / Apertus attention Modules.

**How it works** (copy into `docs/primitive-backward.md` as `## RepeatKV`):

**Forward.** Along the chosen axis, copy each slice `n_rep` times in a row
(`n_rep ≥ 1`). Example: two heads and `n_rep = 3` become six heads
`[h0, h0, h0, h1, h1, h1]`. Cost: 0 FLOPs (copies only). When `n_rep = 1`,
the output is the same memory as the input.

**Backward.** Each original slice was reused `n_rep` times, so its gradient is
the **sum** of those copies (not the average). Cost: *(output elements −
input elements)* additions when `n_rep > 1` and the input needs a gradient;
otherwise 0. Keep nothing (`n_rep` and `axis` are stored on the op).

$$
\frac{\partial L}{\partial A_{i}}
= \sum_{j=0}^{n_{\mathrm{rep}}-1}
\frac{\partial L}{\partial Y_{i \cdot n_{\mathrm{rep}} + j}}
$$

**Broadcast.** This copies one chosen axis on purpose. It does not stretch
size-1 axes the way `Add` does. A grouped-query Module must name the **head**
axis (`axis=1` on a `(batch, n_kv, seq, head_dim)` tensor). Repeating axis 0
of that layout would copy the **batch**, which is the wrong thing.

**Memory.** Store the input's gradient at the input's shape. Do not allocate a
second full-size buffer: the incoming gradient already has the output shape
and is only added together. When `n_rep = 1`, reuse the input's storage
(like `Identity`) and do not allocate a separate input gradient — the output's
gradient *is* the input's gradient.

**Operational contract** (`src/zepto/core/operation/repeat_kv.py`):

```python
"""KV-head replication along one axis."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, numel, persist_only_events, reduced_gradient_metadata
from .records import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)

GRAD_INPUT = "grad_input"


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"repeat_kv axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class RepeatKV(Operation):
    """Repeat the selected axis ``n_rep`` times (GQA KV-head expand)."""

    n_rep: int
    axis: int = 0

    @property
    def family(self) -> str:
        return "repeat_kv"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad and self.n_rep > 1:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.n_rep < 1:
            raise ValueError("repeat_kv n_rep must be >= 1")
        (input_meta,) = inputs
        if not input_meta.shape:
            raise ValueError("repeat_kv requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_meta.shape))
        shape = list(input_meta.shape)
        shape[axis] *= self.n_rep
        return (
            ValueMetadata(
                tuple(shape),
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad,
            ),
        )

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        if self.n_rep == 1:
            return (AliasSpec("input"),)
        return ()

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        input_meta = context.metadata_for("input")
        output = context.metadata_for("output")
        if input_meta is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        if input_meta.requires_grad and self.n_rep > 1:
            return numel(output) - numel(input_meta)
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        if self.n_rep == 1 and result.aliases:
            return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

### 10. `MaterializedCausalMask`

**Family:** `materialized_causal_mask`

**Semantics:** Create a fixed “do not look at the future” table for attention,
shape `(1, seq, seq)`, with no tensor inputs. Position \(i\) may see position
\(j\) only when \(j \le i\): those entries are 0, later positions are
\(-\infty\). Not trainable. 0 FLOPs.

**Why a dedicated Operation**

- Pure **resource declaration** (allocation event) parameterized by `seq_len`.
- Cannot be derived from arithmetic ops on activations — there is no input
  tensor to transform.

**Why not a Module**

- Module composition implies suboperations; here the semantics *are* “create
  persistent/broadcast mask storage.”

**Why not `CausalMask` on scores**

- Atto Apertus uses **materialized additive** mask + `Add`, not structural
  upper-triangle view alone.

**Why not lowering of `Add`**

- The mask buffer must exist as a graph value before scores exist; lowering
  cannot invent a new first-class tensor without a structural producer.

**Unlocks:** Apertus / HF-style GQA (`mask_style="materialized_additive"`).

**How it works** (copy into `docs/primitive-backward.md` as `## MaterializedCausalMask`):

**Forward.** Build one table \(M\) of shape `(1, S, S)` where \(S\) is the
sequence length. For query row \(i\) and key column \(j\): 0 if \(j \le i\)
(allowed), \(-\infty\) if \(j > i\) (blocked). No inputs, no arithmetic
(0 FLOPs). The table is a constant: it is not trained.

$$
M_{0,i,j} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

**Backward.** None. There is nothing to differentiate. Adding this table to
attention scores is a later `Add`.

**Broadcast.** This op does not stretch anything. A later `Add` stretches
`(1, S, S)` across batch and heads when scores are `(batch, heads, S, S)`.

**Memory.** Allocate the table and keep it for the whole run. Keep nothing
else.

**Operational contract** (`src/zepto/core/operation/materialized_causal_mask.py`):

```python
"""Shared additive causal mask allocation."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec
from .base import Operation
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)


@dataclass(frozen=True, slots=True)
class MaterializedCausalMask(Operation):
    """Allocate a persistent additive causal mask of shape ``(1, S, S)``."""

    seq_len: int

    @property
    def family(self) -> str:
        return "materialized_causal_mask"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return ()

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=False)

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.seq_len <= 0:
            raise ValueError("materialized_causal_mask seq_len must be positive")
        return (
            ValueMetadata(
                (1, self.seq_len, self.seq_len),
                semantic_type="tensor",
                requires_grad=False,
                persistent=True,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return (
            ResourceEvent(ResourceEventKind.ALLOCATE, "output:0"),
            ResourceEvent(ResourceEventKind.PERSIST, "output:0"),
        )
```

---

### 11. `Where`

**Family:** `where`

**Semantics:** Per element, pick the “true” value or the “false” value
according to a condition (treated as yes/no). All three inputs are stretched
to a common shape first — the condition counts too, not only the two values.

**Why a dedicated Operation**

- Piecewise activations (xIELU positive/negative branches) cannot be expressed
  from `Add` / `Multiply` alone without a **selection** primitive.
- `Maximum` / `Minimum` cover ReLU-style piecewise linear cases but not
  distinct formulas per branch (xIELU uses different nonlinear forms).

**Why not a Module of `Maximum`/`Minimum`**

- xIELU is not a min/max of two candidate values; it is conditional replacement
  of negative values with `α * f(x)`.

**Why not lowering of `Exp` / `Pow`**

- Lowering chooses implementation; it does not introduce conditional forward
  semantics.

**Why not defer xIELU as fused Operation**

- xIELU is Atto’s **recipe activation**, not atomic math; `Where` keeps xIELU as
  a Module while preserving irreducible selection.

**Unlocks:** `XIELU` Module, future piecewise activations without new op names.

**How it works** (copy into `docs/primitive-backward.md` as `## Where`):

**Forward.** At each position: if the condition is true, take the first value,
otherwise the second. Stretch **all three** inputs to the same shape first
(size-1 axes grow; extra leading axes on the condition also count). Example:
condition `(3, 4)`, true values `(3, 1)`, false values `(1,)` → output
`(3, 4)`, not `(3, 1)`. Cost: 0 FLOPs (picks, no arithmetic). The output needs
a gradient if either *value* does — not if only the condition does.

$$
Y =
\begin{cases}
T & C \text{ is true} \\
F & C \text{ is false}
\end{cases}
$$

**Backward.** The condition is a discrete choice, so it gets no gradient.
Where the condition was true, the incoming gradient goes to the true-value
input (the false side gets 0 there), and the other way around. Stretch the
condition to the output shape before applying that mask. If a value input was
stretched in the forward pass, **add** those extra copies back so the gradient
matches its original shape. Zeros on the unused side still take part in that
sum. Cost: one multiply per output element for each value that needs a
gradient, plus the usual extra additions if that value was stretched.

$$
\frac{\partial L}{\partial T}
= \mathrm{sum\_to\_shape}\!\left(
  \frac{\partial L}{\partial Y} * \mathbb{1}[C],\
  \text{shape of } T
\right)
\qquad
\frac{\partial L}{\partial F}
= \mathrm{sum\_to\_shape}\!\left(
  \frac{\partial L}{\partial Y} * \mathbb{1}[\neg C],\
  \text{shape of } F
\right)
$$

**Broadcast.** Stretch condition, true values, and false values together —
never infer the output shape from the two values alone.

**Memory.** Like `Multiply`: if a value was stretched, first compute a
full-size masked gradient, then sum it down. That temporary exists only for a
stretched value. Keep the condition if either value needs a gradient. Do not
produce a gradient for the condition.

**Operational contract** (`src/zepto/core/operation/where.py`):

On implementation, put the “stretch several shapes together” helper next to
the existing two-input helper. `Where` must use all three inputs to decide
the output shape, not only the two values.

```python
"""Elementwise conditional selection operation declaration."""

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    backward_gradient_port_events,
    broadcast_reduction_flops,
    numel,
    persist_only_events,
    reduced_gradient_metadata,
    unreduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_ON_TRUE = "grad_on_true"
GRAD_ON_TRUE_UNREDUCED = "grad_on_true_unreduced"
GRAD_ON_FALSE = "grad_on_false"
GRAD_ON_FALSE_UNREDUCED = "grad_on_false_unreduced"


def _broadcast_shape(
    shapes: tuple[tuple[int, ...], ...], *, family: str
) -> tuple[int, ...]:
    """NumPy right-aligned broadcast of one or more shapes."""
    rank = max(len(shape) for shape in shapes)
    padded = tuple((1,) * (rank - len(shape)) + shape for shape in shapes)
    result: list[int] = []
    for dims in zip(*padded, strict=True):
        leading = dims[0]
        for dim in dims[1:]:
            if dim == leading:
                continue
            if dim == 1:
                continue
            if leading == 1:
                leading = dim
                continue
            raise ValueError(
                f"{family} shapes are not broadcast-compatible: {shapes}"
            )
        result.append(leading)
    return tuple(result)


def _where_output_metadata(
    condition: ValueMetadata,
    on_true: ValueMetadata,
    on_false: ValueMetadata,
    *,
    family: str,
) -> ValueMetadata:
    shape = _broadcast_shape(
        (condition.shape, on_true.shape, on_false.shape),
        family=family,
    )
    return ValueMetadata(
        shape,
        semantic_type="tensor",
        requires_grad=on_true.requires_grad or on_false.requires_grad,
    )


class Where(Operation):
    """Select ``on_true`` or ``on_false`` per condition element."""

    @property
    def family(self) -> str:
        return "where"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec("condition"),
            PortSpec("on_true"),
            PortSpec("on_false"),
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(GRAD_ON_TRUE, ValueKind.GRADIENT),
            PortSpec(GRAD_ON_FALSE, ValueKind.GRADIENT),
            PortSpec(GRAD_ON_TRUE_UNREDUCED, ValueKind.GRADIENT),
            PortSpec(GRAD_ON_FALSE_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        _condition, on_true, on_false = inputs
        (output,) = outputs
        return (
            reduced_gradient_metadata(on_true),
            reduced_gradient_metadata(on_false),
            unreduced_gradient_metadata(output),
            unreduced_gradient_metadata(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        _condition, on_true, on_false = inputs
        (output,) = outputs
        active: list[str] = []
        if on_true.requires_grad:
            if on_true.shape != output.shape:
                active.extend((GRAD_ON_TRUE, GRAD_ON_TRUE_UNREDUCED))
            else:
                active.append(GRAD_ON_TRUE)
        if on_false.requires_grad:
            if on_false.shape != output.shape:
                active.extend((GRAD_ON_FALSE, GRAD_ON_FALSE_UNREDUCED))
            else:
                active.append(GRAD_ON_FALSE)
        return tuple(active)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("condition",),
            gradient_inputs=("output",),
            gradient_outputs=("on_true", "on_false"),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        condition, on_true, on_false = inputs
        return (
            _where_output_metadata(
                condition, on_true, on_false, family=self.family
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        _condition, on_true, on_false = inputs
        if on_true.requires_grad or on_false.requires_grad:
            return ("condition",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        on_true = context.metadata_for("on_true")
        on_false = context.metadata_for("on_false")
        output = context.metadata_for("output")
        if on_true is None or on_false is None or output is None:
            raise ValueError(
                "Estimation context must provide 'on_true', 'on_false', and 'output' ports"
            )
        flops = 0
        if on_true.requires_grad:
            flops += numel(output)
            if on_true.shape != output.shape:
                flops += broadcast_reduction_flops(on_true, output)
        if on_false.requires_grad:
            flops += numel(output)
            if on_false.shape != output.shape:
                flops += broadcast_reduction_flops(on_false, output)
        return flops

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        unreduced_by_port = {
            GRAD_ON_TRUE: GRAD_ON_TRUE_UNREDUCED,
            GRAD_ON_FALSE: GRAD_ON_FALSE_UNREDUCED,
        }
        active = set(result.active_auxiliary_ports)
        for port_name in (GRAD_ON_TRUE, GRAD_ON_FALSE):
            if port_name not in active:
                continue
            unreduced = unreduced_by_port[port_name]
            if unreduced in active:
                events.extend(
                    backward_gradient_port_events(
                        port_name, unreduced_port=unreduced
                    )
                )
            else:
                events.extend(persist_only_events(port_name))
        return tuple(events)
```

---

## Reclassified: Atto names that must **not** be Operations

These appeared in the earlier Atto parity gap list but **fail the irreducibility
test**. Build them as **Modules** (and optionally match with **fused
Implementations** at lowering).

| Atto / suggested name | Zepto approach | Reason |
|----------------------|----------------|--------|
| **ReLU** | `Maximum(x, 0)` + `maximum/relu-mask` lowering | Piecewise linear max with zero; backward saves post-activation output via Maximum contract fix (`PLAN.md` §3.1) |
| **Softmax** | Module: `Exp` → `ReduceSum` → `Divide` (broadcast) | Classic 3-step composition; Atto folds scale into one node for FLOP convenience, not atomicity |
| **CausalMask** (structural) | Module flag on attention + lowering (`masked_softmax`) OR skip if only materialized path needed | Shape-preserving mask view; no arithmetic; Apertus uses materialized path instead |
| **LayerNorm** | `zepto.modules.LayerNorm` Module | `PLAN.md` §6: compose reduce, pow, sqrt, add, multiply |
| **RMSNorm** | `zepto.modules.RMSNorm` Module | Same; Atto primitive is a **cost leaf**, not irreducible math |
| **RoPEMaterialize** | `zepto.modules` or recipe submodule | `MatMul(inv_freq, position_ids)` + `Sin`/`Cos` + scale — Atto already documents decomposed FLOPs |
| **RoPE** (apply) | Module over `Split`/views + `Sin`/`Cos`/`Multiply`/`Add` | Rotation is algebraic composition on half-dim pairs |
| **XIELU** | Module over `Where`, `Exp`, `Pow`, `Multiply`, `Add` + parameter ports | Trainable scalars are parameters, not a new op family |
| **EmbeddingLookup** | `Embedding` Module calling `Gather` | User-facing recipe; op layer is `Gather` |
| **CrossEntropy** | Training Module over `Log`, `Softmax` Module, `Gather` | Loss is a graph region, not a unary/binary primitive |
| **GeLU / SiLU / SwiGLU** | Modules (future) | Documented in Atto framework as activation leaves; SwiGLU needs extra GEMM + gate, i.e. recipe |

### Lowering / fusion (not new Operations)

| Pattern | Approach |
|---------|----------|
| Fused LayerNorm / RMSNorm | Region matcher on module provenance + composite `Implementation` |
| Fused masked softmax | `Softmax` Module region + causal metadata in `InvocationContext` |
| ReLU | Existing `Maximum` + specialized lowering |
| FlashAttention / SDPA | Attention **Module** region lowering; no new structural op |

---

## Coverage check: Atto recipes with atomic ops + Modules

```text
Apertus
├── Embedding Module          → Gather + parameters
├── MaterializedCausalMask    → Operation #10
├── RoPE Module               → Sin, Cos, Exp, MatMul, Gather (position ids)
├── RMSNorm Module            → ReduceSum, Pow, Add, Sqrt, Multiply + γ param
├── GQA Module                → MatMul, Reshape, Transpose,
│                               RepeatKV(n_rep, axis=1) (#9),
│                               Add, Softmax Module, MatMul
├── XIELU Module              → Where (#11), Exp, Pow, Multiply, Add + α params
└── residuals                 → Add (existing)

VaswaniDecoder
├── Embedding Module          → Gather
├── Sinusoidal PE             → persistent tensor + Add (no new op)
├── LayerNorm Module          → ReduceSum, Pow, Sqrt, Add, Multiply + γ, β
├── MHA Module                → same attention ops as GQA without RepeatKV
│                               (n_kv = n_heads); causal via mask config
├── FFN Module                → MatMul, Maximum(x,0) [ReLU], MatMul
└── CrossEntropy Module       → Log, Softmax Module, Gather (train)
```

---

## Implementation order

1. **Parameter ports** (prerequisite) — weight-bearing Modules cannot bind until
   `Operation.parameter_ports` and `context.parameter()` work.
2. **Layout / IO:** `Concat`, `Gather`, `RepeatKV`, `MaterializedCausalMask`
3. **Unary / reduction math:** `ReduceSum`, `Pow`, `Exp`, `Log`, `Sin`, `Cos`,
   `Where`
4. **Modules:** `RMSNorm`, `LayerNorm`, `Softmax`, RoPE pair, `Embedding`,
   `XIELU`, attention (MHA/GQA), `CrossEntropy`
5. **Lowering:** fused norm, masked softmax, ReLU-on-Maximum (partially exists)
6. **Docs:** for each shipped op, copy its **How it works** section into
   [docs/primitive-backward.md](docs/primitive-backward.md) as a heading. That
   file is then the source of truth for the explanation.

---

## Summary table

| # | Operation | Family | Atto primitive replaced by decomposition |
|---|-----------|--------|------------------------------------------|
| 1 | ReduceSum | `reduce_sum` | (elementary — enables Softmax, norms) |
| 2 | Pow | `pow` | (elementary) |
| 3 | Exp | `exp` | (elementary) |
| 4 | Log | `log` | (elementary — enables CE) |
| 5 | Sin | `sin` | part of RoPEMaterialize / RoPE |
| 6 | Cos | `cos` | part of RoPEMaterialize / RoPE |
| 7 | Concat | `concat` | Concat |
| 8 | Gather | `gather` | EmbeddingLookup |
| 9 | RepeatKV | `repeat_kv` | RepeatKV |
| 10 | MaterializedCausalMask | `materialized_causal_mask` | MaterializedCausalMask |
| 11 | Where | `where` | (elementary — enables XIELU) |

**11 atomic operations** to add. **14 Atto-named IR ops** from the prior gap analysis
collapse to these 11 plus **Modules** and **lowering**, consistent with Zepto’s
domain model and `PLAN.md` §6.
