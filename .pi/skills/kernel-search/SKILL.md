---
name: kernel-search
description: >
  Research a neural-network operation or GPU kernel for Zepto cost modeling:
  math definition, FLOPs (forward/backward), HBM/VRAM, fusion boundaries,
  implementation catalog, and YAML recipe spec. Use when analyzing kernels
  (RMSNorm, Softmax, GQA/FlashAttention, RoPE, xIELU, linear+CE, etc.)
  for RegionImplementation or docs/kernel-implementation.md.
compatibility: Requires network access for web search and GitHub/source lookup.
metadata:
  version: "0.2"
  output: _workspace/research.md
---

# kernel-search

Produce a **Zepto implementation research report** for a requested operation or kernel family.

**Trigger:** `/skill:kernel-search <operation>` or natural language (e.g. “Analyze masked softmax for Zepto”).

**Output:** write the complete report to `_workspace/research.md` using `assets/report-template.md`.

---

## Inputs (extract or ask)

- **Target operation** (e.g. `RMSNorm`, `Softmax`, `MaskedSoftmax`, `GQA/FlashAttention`, `FusedLinearCE`, `RoPE`, `xIELU`)
- **Scope:** standalone leaf vs full attention vs LM-head (fusion boundary A/B/C/D)
- **Estimation context:** prefill vs decode, training vs inference, hardware (CUDA / ROCm / XPU / MPS)

**Authority split (mandatory):**

- Checkpoint configs + library source → *what* is computed
- Papers → *why* a fused kernel has given FLOP / memory complexity
- HuggingFace eager Python → semantic reference
- Liger / FlashAttention / vLLM / Hub wheels → execution strategies modeled as alternative **implementations**

---

## Research workflow (default order)

1. **Read local authority first:**
   - `docs/kernel-implementation.md` (existing section if any)
   - `src/zepto/analysis/lowering/implementations/regions/`
   - `CONTEXT.md` — cost conventions
2. **Web search** for primary sources (stop when each implementation row has a URL):
   - HuggingFace Transformers eager path (GitHub, line refs)
   - Liger / FlashAttention / Hub kernel repo
   - Primary paper (arXiv)
3. **Fill** `_workspace/research.md` from `assets/report-template.md`.
4. **Load detailed section templates** from `references/report-sections.md` while writing.
5. **Validate** before finishing:
   ```bash
   python3 .pi/skills/kernel-search/scripts/validate_research.py _workspace/research.md
   ```
   Fix all failures and re-run until pass.

---

## Cost conventions

Follow `CONTEXT.md` and `docs/kernel-implementation.md` §1. For the full notation table and rules, read `references/cost-conventions.md`.

---

## Gotchas

Read `references/gotchas.md` before writing. Critical defaults:

- Default Zepto leaf = **kernel-accurate**, not Appendix E paper-comparable
- Stable row softmax leaf: max + sub + exp + sum + div = **5× |tile|**
- Never mix theoretical FLOPs with HBM traffic in one number
- Identity lowering resource chains are mandatory even when recommending fused regions
- Structural causal mask (`is_causal=True`) → **0 mask bytes**
- SRAM/SLM/threadgroup temps → omit from `ALLOCATE`/`SAVE` chains
- `requires_grad=False` → backward FLOPs = 0
- Do not claim wall-clock speedups; use HBM traffic and peak VRAM

---

## Section scope

Not every operation needs every section. Apply this gate; write **TBD** with a search plan only when a required section lacks data.

| Operation type | Required sections | Optional / cross-ref |
|----------------|-------------------|----------------------|
| Standalone norm/activation (RMSNorm, xIELU, ReLU) | 0, 1, 4, 5, 6, 8 | Full attention catalog (§2 subset OK) |
| Attention softmax sub-leaf (boundary A/B) | 0, 1, 3, 4, 5, 6, 8 | Boundary C/D cross-ref |
| Full GQA / FlashAttention (boundary C) | All (0–9) | — |
| LM-head fused CE (boundary D) | 0, 1, 2, 3, 4, 5, 6, 8 | Attention sections |
| Inference-only, no grad | 0, 1, 4, 6, 8 | Skip §5; state backward = 0 |

Section numbers and templates: `references/report-sections.md`.

---

## Reference files (load on demand)

| File | When to read |
|------|--------------|
| `references/report-sections.md` | Writing the report (section templates) |
| `references/cost-conventions.md` | Notation, FLOP rules, Apertus defaults |
| `references/gotchas.md` | Before costing; after failed eval |
| `references/implementation-catalog.md` | Building §2 backend table |
| `references/yaml-recipe-schema.md` | Writing §8 YAML spec block |

---

## System rules

1. Step tables first, closed form second — never abbreviate FLOP derivations.
2. Always separate forward FLOPs, backward FLOPs, forward-lived VRAM, saved activations, and resource event chains.
3. Always distinguish transient allocations vs saved-for-backward vs persistent weights.
4. Always report paper-comparable vs kernel-accurate FLOPs when they differ.
5. Cite primary sources — no uncited FLOP constants.
6. Flag known Zepto gaps when applicable (**G1** RoPE θ, **G3** KV cache, **G4/G4b** unfused GQA false peak).

---

## Quality checklist (self-verify before finishing)

- [ ] Mathematical definition with shapes (§0)
- [ ] Three-layer costing model named (§1)
- [ ] Fusion boundaries A/B/C/D classified where applicable (§3)
- [ ] Forward FLOP step table + closed form (§4)
- [ ] Backward FLOP step table + closed form, or backward = 0 noted (§5)
- [ ] Paper vs kernel-accurate FLOPs reconciled
- [ ] Arithmetic intensity with numeric example (§4 or §6)
- [ ] Forward-lived VRAM table (§6.1)
- [ ] Saved activation table per backend variant (§6.2)
- [ ] Resource event chains: identity **and** fused (§6.3)
- [ ] Apertus-8B numerical trace
- [ ] YAML recipe spec with `elided_temps` and `save_*` flags (§8)
- [ ] Composition / mutual-exclusion rules with region ids (§3)
- [ ] Sources + backlog row if not yet registered (§9)
- [ ] `validate_research.py` passes

---

## Evals

Test cases live in `evals/evals.json`. Run evals in fresh sessions; save outputs under a gitignored workspace (see `evals/README.md`).
