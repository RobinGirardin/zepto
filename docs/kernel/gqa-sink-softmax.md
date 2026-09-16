# GQA with sink softmax (GPT-OSS boundary C)

**Status:** archived kernel research  
**Region:** `region/gqa-sink`  
**Reference:** Megatron `SoftmaxOne`; GPT-OSS per-head sink denominator

## Math

Standard attention softmax over scores \(S \in \mathbb{R}^{h \times S \times S}\) adds a per-query-head sink scalar to the denominator (equivalent to an extra key slot absorbing probability mass). Zepto models this as one fused op `attention_softmax_with_sink` rather than decomposed `exp` / `reduce_sum` / `divide`.

## Fusion boundary C (7 ops)

```
repeat_kv → repeat_kv → transpose → matmul → add → attention_softmax_with_sink → matmul
```

Mutually exclusive with the 10-op standard GQA pattern (no `exp` family in the sink path).

## Forward FLOPs

Let \(N = N_{\mathrm{pairs}}(S, W)\) be effective causal + sliding-window pairs (see `effective_attention_pairs` in `recipes/gqa.py`):

\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 4 h N d_h + 5 h N + h S
\]

- \(4 h N d_h\): QK + PV GEMMs  
- \(5 h N\): online softmax over allowed pairs  
- \(h S\): sink denominator term per `AttentionSoftmaxWithSink` convention

## VRAM

Same elision as `region/gqa` flash variants:

- No materialized \((h, S, S)\) attention temps  
- Optional `row_stats` save \(\Theta(h S)\) when autograd requires backward  
- `MaterializedCausalMask` / `MaterializedSlidingWindowCausalMask` storage elided when flash-class backend fuses the core (see `docs/kernel-implementation.md` §10–§12)

## Variants

| Id | Notes |
|----|-------|
| `region/gqa-sink/flash2` | Default on non-CUDA hardware |
| `region/gqa-sink/flash3` | CUDA-only, wins on `hardware=cuda` |
