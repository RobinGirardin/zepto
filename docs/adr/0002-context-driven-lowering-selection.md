---
status: accepted
---
# Select implementations through context-driven lowering

Structural operations remain backend-neutral. A single explicit lowering registry contains separate implementation descriptors with stable identities, typed capabilities, priorities, resource behavior, and lowering logic; lowering selects among compatible descriptors using the immutable estimation context. Users may express operation- or context-level family preferences, exact pins, and hard requirements, with conflicting hard requirements failing explicitly and fallback requiring explicit permission.

This avoids coupling model declarations to CUDA, MPS, XPU, or a particular library while still supporting reproducible research estimates and automatic selection. Selection records retain rejected candidates and reasons so a report explains why an implementation was chosen.
