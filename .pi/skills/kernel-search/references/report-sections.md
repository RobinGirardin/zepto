# Report section templates

Produce sections **in order** (0–9). Write **TBD** with a search plan if a required fact is unknown.

---

## Section 0: Mathematical definition

1. Closed-form definition of the operation (LaTeX).
2. Input/output tensor shapes and reduction axes.
3. Numerics policies that affect **bytes** but not FLOPs (e.g. HF fp32 softmax tile, stable max-subtract).
4. Structural-graph meaning vs eager decomposition (which ops fuse).

---

## Section 1: Three-layer costing model

For this operation, name all three Zepto layers:

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | e.g. `aten::softmax`, Liger `LigerRMSNorm`, FlashAttention-2 |
| **Identity lowering** | Structural / debug chain | Zepto primitive sequence (cite `src/zepto/modules/…` if present) |
| **Zepto fused region** | Kernel-accurate cost leaf | e.g. `region/rmsnorm/liger`, `region/masked_softmax` |

**Default estimates use the reference fused kernel or its Zepto region — not a sum of identity primitives.**

---

## Section 2: Implementation catalog & taxonomy

See `implementation-catalog.md` for minimum backend coverage and table format.

---

## Section 3: Fusion boundary & composition rules

Classify the operation using boundaries A/B/C/D (see `cost-conventions.md`).

Document:

- **Mutually exclusive** region ids that replace each other
- **Composable** stacks (what fuses sequentially)
- **Execution constraints:** layouts, dtypes, causal mask materialization vs structural, hardware gates

---

## Section 4: Forward FLOPs — step-by-step derivation

**Mandatory:** derive forward FLOPs in a **step table**, not a single formula.

For each billed stage list: primitive name, work tile shape, FLOPs per element or reduction rule, running subtotal.

Produce **separate subtotals** for:

1. **Identity lowering** (unfused Zepto chain today)
2. **Reference fused kernel / Zepto region leaf** (kernel-accurate)
3. **Paper / Appendix E comparable** (if different — explain why)
4. **Full parent op** if this is a sub-leaf (e.g. softmax inside GQA adds GEMMs separately)

**Stable softmax rule:** fused leaf bills **max + subtract + exp + sum + div = \(5 \cdot |tile|\)** per row-softmax tile.

**Reduction billing:** row reductions bill like RMSNorm — one FLOP-equivalent per element of the reduction output.

Close with closed-form:
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = \cdots
\]

**Arithmetic intensity:** include a numeric example (FLOPs / minimum HBM bytes) using Apertus-8B defaults where applicable.

---

## Section 5: Backward FLOPs — step-by-step derivation

**Mandatory separate section** for training contexts. Do not fold into forward.

For each path document: what is **saved** vs **recomputed**, backward step table, closed-form:
\[
\mathrm{FLOPs}_{\mathrm{bwd}} = \cdots
\]

**Common patterns:**

| Pattern | Saved | Backward FLOP effect |
|---------|-------|----------------------|
| Save output \(P\) | Full softmax output | \(\approx 4|S|^2\) for stable softmax |
| Save pre-softmax \(z\) | Logits tile | Same order as \(P\) path |
| Save row stats \((m,\ell)\) only | FlashAttention-style | Lower saved memory, higher backward FLOPs |
| Recompute from \(x\) only | MPS mlx-rmsnorm inference | No `rstd` HBM; backward recomputes reduction |

State `requires_grad=False` → backward FLOPs = 0.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers (name + shape) | Peak bytes | Elided by fusion |
|------|----------------------------|------------|------------------|

List **every** full-rank temp in identity lowering.

For chunk/streaming kernels (Liger linear CE):
\[
\mathrm{peak} = \min(SVe,\; C \cdot S \cdot d \cdot e)
\]
with documented chunk constant \(C\) (Liger: **16**).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes (order) | Zepto `SAVE` policy |
|------|---------------|-------|---------------|---------------------|

Distinguish persistent weights / causal mask, forward-lived temps, saved-for-backward auxiliaries.

### 6.3 Resource event chains (mandatory)

For **each** costing path, write ordered Zepto **resource event chains** (identity vs fused side-by-side).

See `yaml-recipe-schema.md` for event kinds and examples.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|

---

## Section 8: Zepto implementation spec (recipe fields)

Emit the YAML block per `yaml-recipe-schema.md`.

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Papers (arXiv links)
- Library sources (GitHub paths: Transformers, Liger, FlashAttention, Megatron, Hub repos)
- Verification date

### 9.2 Zepto backlog row (if `status: to_implement`)

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|

Mirror format of `docs/kernel-implementation.md` backlog sections.
