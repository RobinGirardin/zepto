# Zepto kernel research: {{OPERATION}}

**Date:** {{DATE}}
**Scope:** {{SCOPE}}
**Context:** {{CONTEXT}}

---

## Section 0: Mathematical definition

<!-- LaTeX definition, I/O shapes, numerics policies, structural vs eager -->

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| Reference fused kernel | | |
| Identity lowering | | |
| Zepto fused region | | |

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|

---

## Section 3: Fusion boundary & composition rules

**Boundary:** <!-- A | B | C | D -->

**Mutually exclusive:**

**Composable:**

**Execution constraints:**

---

## Section 4: Forward FLOPs — step-by-step derivation

### 4.1 Identity lowering

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|

### 4.3 Paper-comparable (if different)

<!-- Label explicitly; default Zepto leaf is kernel-accurate -->

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = \cdots
\]

**Arithmetic intensity (numeric example):**

---

## Section 5: Backward FLOPs — step-by-step derivation

<!-- Or: requires_grad=False → backward FLOPs = 0 -->

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} = \cdots
\]

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes | Elided by fusion |
|------|--------------|------------|------------------|

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
<!-- ALLOCATE → … → SAVE -->
```

**Fused region leaf:**
```
<!-- ALLOCATE → … → SAVE -->
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|

---

## Section 8: Zepto implementation spec

```yaml
region_kind:
recommended_region_ids:
  - id:
    variant:
    hardware_gate:
    fusion_boundary:
    status:
pattern_rule:
  op_families: []
recipe:
  forward_flops:
  backward_flops:
  elided_temps: []
  saved_backward: []
  resource_events_forward: []
  numerics_tags: []
capabilities: []
priority:
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- 

**Verification date:**

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
