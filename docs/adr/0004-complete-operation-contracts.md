---
status: accepted
---
# Require complete semantic and estimation operation contracts

Every structural operation must provide both semantic and estimation protocols at declaration time. The composed contract covers ports, symbolic shape/type behavior, parameter and alias semantics, forward/backward behavior, saved state, resource behavior, and theoretical FLOPs using the two-FLOP multiply-add convention; incomplete custom operations are rejected eagerly.

This keeps identity lowering and cost estimation available for research-defined operations immediately, while allowing specialized implementations to replace execution/resource behavior only through explicit mappings that preserve structural meaning.
