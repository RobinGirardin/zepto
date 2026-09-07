# Cost conventions

Follow Zepto / [`CONTEXT.md`](../../../../CONTEXT.md) and `docs/kernel-implementation.md` §1.

## Rules

| Rule | Detail |
|------|--------|
| FLOP convention | One multiply-add = **2 FLOPs** |
| Two costs — never mix | (1) **Theoretical FLOPs** (closed form); (2) **HBM traffic + peak allocated VRAM** (bytes live / moved) |
| Paper vs kernel-accurate | **Appendix E / paper-comparable** formulas may differ from **kernel-accurate Zepto leaf** (e.g. softmax \(3hS^2\) paper vs **\(5hS^2\)** stable fused leaf) — report **both** when they diverge |
| SRAM vs HBM | On-chip register/SLM temps → **no** `ALLOCATE`/`SAVE` events |
| VRAM unit | \(1\,\mathrm{GiB} = 2^{30}\) bytes |
| Default numerics example | Apertus-8B: \(S{=}8192\), \(d{=}4096\), \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\), \(V{=}131072\), bf16 \(e{=}2\) |

## Notation table

Define every symbol used in the report:

| Symbol | Meaning | 8B example |
|--------|---------|------------|
| \(S\) | sequence length (prefill) | 8192 or 65536 |
| \(d\) | hidden size | 4096 |
| \(d_{\mathrm{ff}}\) | MLP width | 21504 |
| \(h\) | query heads | 32 |
| \(h_{\mathrm{kv}}\) | key/value heads | 8 |
| \(d_h\) | head dim | 128 |
| \(L\) | layers | 32 |
| \(V\) | vocab size | 131072 |
| \(e\) | element size in bytes | 2 (bf16), 4 (fp32) |
| \(B\) | batch size | context-dependent |
| \(M\) | on-chip SRAM bytes per SM | hardware-specific |

## Fusion boundaries

| Boundary | Meaning | Example regions |
|----------|---------|-----------------|
| **A** | Standalone row softmax / norm | `region/softmax` |
| **B** | Scale + mask + softmax (scores only) | `region/masked_softmax` |
| **C** | Full attention (QKᵀ + softmax + PV) | `region/gqa/*` |
| **D** | LM-head GEMM + vocab softmax + CE | `region/linear_ce` |

## Registered Zepto regions (cross-check)

`region/linear`, `region/layernorm`, `region/relu`, `region/rmsnorm`, `region/xielu`.
