---
status: accepted
---
# Use explicit tensor metadata and separate memory accounting

Zepto keeps three concerns distinct:

- `ValueKind` describes how a port participates in an operation contract.
- `semantic_type` describes the mathematical or domain meaning of a tensor.
- tensor accounting metadata describes dtype, storage role, gradient
  participation, and persistence.

Tensor and parameter declarations may provide explicit accounting metadata.
Context policies provide defaults only where declarations do not provide an
override. Explicit tensor or parameter dtype takes precedence over operation or
port overrides, which take precedence over role-based defaults and finally
context-wide defaults. Trainability belongs to parameters; `requires_grad`
describes gradient participation for graph values.

Saved backward values are ordinary tensors. They may use a semantic type such
as `auxiliary_state`, and their lifetime is described through ordinary tensor
metadata and allocation, persistence, and release behavior. They are not
special opaque memory records. They need not be public module outputs: an
operation may expose them as internal graph values while returning only its
declared public outputs.

Operations distinguish public outputs from internal tensor values through
separate output and auxiliary port declarations. Auxiliary values are
registered in the structural graph, receive tensor and storage identities, and
can be referenced by backward metadata. Functional wrappers return only public
outputs.

Operation validation is organized into dedicated declaration, invocation, and
graph validators. Declaration validation checks the reusable contract,
invocation validation checks concrete inputs and inferred results, and graph
validation checks ownership, ordering, bindings, storage, and graph outputs.
These validators remain behind the public operation and graph validation
methods and do not perform invocation-specific memory accounting during
structural graph construction.

Resource events do not have a dedicated gradient kind. Gradient tensors and
gradient accumulation are represented through tensor and port metadata plus
ordinary allocation, persistence, and release events. A specialized event may
be added later only if those events cannot express a demonstrated requirement.

Lowering and accounting are separate operations:

```text
structural graph → lower → lowered graph
                         ├→ account_memory → MemoryReport
                         └→ account_flops   → FlopReport
```

`estimate` may compose these operations as a convenience API, but it does not
replace the separate APIs. `MemoryReport` includes both total allocation
(`sum-all`) and peak live allocation, along with persistent, workspace, and
other useful breakdowns.

Optimizer behavior is supplied through an optimizer-policy contract rather
than being embedded in operation declarations. The policy describes optimizer
state tensors, their precision and persistence, and update-time resource
behavior.

Horizon FLOP accounting for optimizer steps uses ``OptimizerPolicy.flops_per_parameter``
times trainable parameter element count rather than summing lowered model nodes.
The optimizer timeline row carries no model compute nodes; update cost is attributed
under ``optimizer/<policy name>`` in flop reports.

