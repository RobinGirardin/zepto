# Zepto kernel-search gotchas

Corrections agents get wrong without explicit guidance. Add new entries when evals fail repeatedly.

## FLOP accounting

- **Default Zepto leaf = kernel-accurate**, not Appendix E paper-comparable. Label paper formulas explicitly when both appear.
- **Stable row softmax:** fused leaf bills max + subtract + exp + sum + div = **\(5 \cdot |tile|\)**. Do not use \(3|S|^2\) as the Zepto leaf unless labeled “paper-comparable only”.
- **Reduction billing:** row reductions (max, sum) bill like RMSNorm — one FLOP-equivalent per element of the reduction output dimension, not silently dropped.
- **RMSNorm fused leaf:** \(4n\) forward FLOPs (square, mean, normalize multiply, scale). Identity lowering may sum to \(4n + 2S\); fused leaf keeps \(4n\).
- **GEMM:** multiply-add = 2 FLOPs (\(2 \cdot m \cdot n \cdot k\)).

## Memory and resource events

- **Never mix** theoretical FLOPs with HBM traffic or peak VRAM in one number.
- **SRAM/SLM/threadgroup temps** → omit from `ALLOCATE`/`SAVE` chains. Only HBM-resident boundary outputs and explicit saved-backward tensors get events.
- **Identity lowering chains are mandatory** even when recommending a fused region — they define what fusion elides.
- **Structural causal mask** (`is_causal=True`, Flash/SDPA) → **0 mask bytes** and **0 mask-add FLOPs**. Do not assume materialized \((1,S,S)\) unless modeling eager HF.
- **Unfused GQA false peak (G4b):** naive sum of simultaneous \((h,S,S)\) temps overstates peak; fusion elides them.

## Backward

- **`requires_grad=False`** → backward FLOPs = 0; skip §5 tables or note explicitly.
- **Save output \(P\)** vs **save row stats \((m,\ell)\) only** (FlashAttention): lower memory, higher backward FLOPs from recomputation.
- **MPS mlx-rmsnorm inference:** `rstd` stays in threadgroup SLM; backward recomputes from \(x\) only.

## Outputs Zepto does not model

- Do not claim wall-clock speedups. Use HBM traffic and peak VRAM.
- RoPE wrong \(\Theta\) (G1) changes numerics, not shapes or FLOPs — use checkpoint frequencies.

## Known Zepto gaps

| Id | Issue |
|----|-------|
| **G1** | RoPE θ / Llama3 scaling from checkpoint vs paper |
| **G3** | KV cache decode accounting |
| **G4/G4b** | Unfused GQA false peak from stacked \((h,S,S)\) temps |

Flag these in the report when the operation touches them.
