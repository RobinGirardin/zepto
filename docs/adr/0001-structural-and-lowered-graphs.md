---
status: accepted
---
# Separate structural and lowered graphs

Zepto will keep an immutable, backend-neutral structural operation graph as the semantic source of truth and derive immutable lowered invocation graphs for concrete contexts. This preserves PyTorch-like eager composition and provenance while allowing backend-specific implementations, fusion, decomposition, aliases, storage events, and future kernel subtypes without conflating model meaning with execution strategy.

The initial lowering pass is identity lowering, but its mappings are explicit at operation-port and tensor-port level so one-to-many and many-to-one transformations remain supported.
