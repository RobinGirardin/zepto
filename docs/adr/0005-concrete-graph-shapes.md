---
status: accepted
---
# Rebuild structural graphs for concrete shape scenarios

Structural graphs use concrete tensor dimensions. Supporting symbolic
dimensions would require shape bindings, constraint validation, invocation-time
resolution, and symbolic cost expressions, while graph construction is
inexpensive for the intended workloads.

When a scenario changes batch size, sequence length, or another shape, Zepto
rebuilds the structural graph with the new concrete input metadata. This keeps
operation inference, FLOP estimation, resource accounting, and graph
validation deterministic. Symbolic dimension support may be reconsidered if
graph construction becomes a measurable bottleneck.
