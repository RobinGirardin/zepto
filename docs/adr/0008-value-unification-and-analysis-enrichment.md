---
status: accepted
---
# Value unification and analysis enrichment

ADR-0007 left `ValueMetadata` as an internal inference type and kept bound port metadata on nodes. That duplicated shape and dtype across compose `Tensor`, graph `Edge.tensor`, port bindings, and `OperationResult.outputs`, and it let lowering and estimation infer accounting roles independently.

Zepto now uses one structural value type (`Tensor` / `Parameter`), one declaration contract type (`PortContract` on `Port.contract`), and one analysis enrichment path (`RoleContext` → `infer_accounting_role` → `ResolvedValue` / `LoweredEdge.role`). Role is not a compose-`Tensor` field. Graph data flow stays edge-based: downstream ops read `Edge.tensor` via edge ids, never node result copies.

## Final types

- **Compose / graph:** `Tensor`, `Parameter`; `Edge.tensor`; `Graph.parameters: Mapping[ParameterId, Parameter]`; `Node` holds wiring plus semantics-only `OperationResult`.
- **Semantic:** `PortContract` (partial specs; unset and empty shape `()` are wildcards); `contract_satisfied(contract, tensor)`; `OperationResult` keeps aliases, saved-for-backward, and active auxiliary names — not output tensors.
- **Analysis:** `RoleContext`, `infer_accounting_role`, `ResolvedValue(tensor, role, dtype)`, `LoweredEdge.tensor` + `.role`, `EstimationContext.port_values`.

## Removals

`ValueMetadata`, `metadata_compatible`, `to_metadata` / `from_metadata`, `Edge.metadata`, `ParameterAsset`, `bound_parameter_metadata`, `OperationResult.outputs` / `auxiliary_outputs`, `EstimationContext.port_metadata` / `metadata_for`.

## Considered options

Keeping `ValueMetadata` as a port-only DTO was rejected: it still duplicated `Tensor` fields and invited role to leak into composition. Putting role on compose `Tensor` was rejected so structural graphs stay backend-neutral and estimation/lowering share one role pipeline.
