# Implementation catalog (Section 2)

Catalog **every** relevant backend in a table:

| Implementation | Package / access | Fused? | Fusion boundary (A/B/C/D) | Device routing | Training vs inference |
|----------------|------------------|--------|----------------------------|----------------|----------------------|

## Minimum coverage

1. **HF Transformers eager** — line-by-line Python path (GitHub file + function names)
2. **PyTorch native** — `aten::`, SDPA dispatcher backends
3. **Liger Kernel** — `liger-kernel`, monkey-patch / `use_liger_kernel=True`
4. **HF Hub Kernels** — `kernels-community/<op>`, `KernelConfig`, `use_kernels=True`
5. **Vendor / C++** — FlashAttention, Megatron/TE, Apex, vLLM, AITER, Metal-Flash, Sage, …

## Per-variant notes

For each Zepto-relevant variant, record:

- Hub repo id and **primary source URL** (GitHub path or HuggingFace repo)
- Whether Transformers routes it by device (CUDA vs ROCm vs XPU vs MPS)
- Inference-only vs training autograd surface
- What tensors are saved for backward (if training)

## Search stop condition

Stop web search when every row in the catalog table has at least one primary source URL. Do not invent package names or FLOP constants without citation.

## Composition (Section 3 cross-ref)

Document **mutually exclusive** region ids (pick one per layer/head) and **composable** stacks (e.g. Liger RMSNorm + Flash Hub + Liger linear CE).

Execution constraints to note: layouts, dtype policies, causal mask materialization vs structural, hardware gates (Hopper FA3, ROCm AITER, MPS inference-only hub paths).
