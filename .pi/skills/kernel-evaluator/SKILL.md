---
name: kernel-evaluator
description: Evaluates and breaks down deep learning layer/operation implementations (e.g. RMSNorm, SwiGLU, RoPE, FlashAttention, CrossEntropy) across standard PyTorch, Liger Kernel, and Hugging Face Hub Kernels. Computes theoretical FLOPs, SRAM/VRAM memory mechanics, autograd saved activations, and hardware compatibility rules.
---

# Operation & Kernel Evaluator

You are performing a comprehensive hardware-level and software-level analysis of a requested Neural Network Operation/Layer. 

When the user calls this skill (e.g., via command `/skill:kernel-evaluator <operation_name>` or in open-ended text like "Analyze RoPE kernels"), extract or prompt for the **Target Operation** (e.g., `RMSNorm`, `SwiGLU`, `RoPE`, `CrossEntropy`, `FlashAttention`, `FusedLinear`).

---

## EXECUTION WORKFLOW

Execute the analysis following this exact 6-part framework:

### Section 1: Implementation Catalog & Taxonomy
Provide a breakdown of how **$ARGUMENTS** (or the target operation) is implemented across the PyTorch & Hugging Face ecosystem:
1. **Standard PyTorch Native / Eager Implementation:** How Python/PyTorch executes it sequentially line-by-line.
2. **Ecosystem Fused-Kernel Implementations:**
   - **Liger Kernel (`liger-kernel`):** Triton training-centric optimizations (monkey-patching / `use_liger_kernel=True`).
   - **Hugging Face Hub Kernels (`kernels-community/<op>`):** Dynamic loader via `KernelConfig` and `use_kernels=True`.
   - **Hardware/Vendor Native Kernels:** E.g., FlashAttention, NVIDIA Apex, vLLM/TGI C++ bindings, or PyTorch native backends.

### Section 2: Implementation & Ecosystem Differences
Compare the fused options (e.g., Liger vs. Hub Kernels vs. Custom C++ CUDA):
- **Delivery & Integration Strategy:** In-place runtime monkey-patching vs. dynamic Hub binary loading vs. package bindings.
- **Primary Optimization Target:** Training backward-pass memory minimization vs. inference runtime latency & multi-hardware portability.
- **Multi-Hardware Support:** CUDA, AMD ROCm, Intel XPU, Apple MLX, etc.

### Section 3: Kernel Interoperability & Composition Rules
- **Layer Scope Overlap Rules:** State clearly which kernels run sequentially (non-overlapping) and which collide (overlapping).
- **Execution Constraints:** Highlight required tensor layouts (memory strides/contiguity), precision requirements (FP16/BF16/FP32), and hardware microarchitecture constraints.

### Section 4: Theoretical FLOP & Memory Bottleneck Analysis
1. **FLOP Derivation:** Compute step-by-step theoretical FLOPs per token:
   - Forward Pass FLOPs formula (using terms $N$ tokens, $D$ hidden dim, $S$ sequence length, or $V$ vocab size).
   - Backward Pass FLOPs formula.
2. **Arithmetic Intensity ($FLOPs / Byte$):**
   - Classify as **Memory-Bound** (HBM Bandwidth restricted) or **Compute-Bound** (Tensor Core restricted).
   - Explain why fused implementations outpace eager execution despite having identical theoretical FLOP counts.

### Section 5: Memory Management (Eager vs. Fused)
Contrast VRAM dynamics across execution modes:
- **Intermediate Allocations (Forward Pass):** Transient VRAM buffers allocated in HBM during eager execution vs. 0-allocation SRAM register execution in fused kernels.
- **Saved Activation Footprint (Autograd):** Full input/output tensor storage in VRAM vs. single-scalar/minimal state saving with SRAM recomputation during backpropagation.
- **Numerical Example:** Provide a concrete VRAM comparison scaled to an LLM architecture (e.g., $D=4096$, $S=4096$, $B=4$).

### Section 6: Summary Table
Conclude with a scannable Markdown summary table:

| Implementation | Package / Access Method | Fused Kernel? | Primary Target | Memory Strategy |
| --- | --- | --- | --- | --- |

---

## SYSTEM PROMPT / MODEL RULES
- Focus heavily on concrete hardware primitives (HBM, L1/L2 Cache, SRAM, Streaming Multiprocessors, Tensor Cores).
- Do not abbreviate the mathematical derivation of theoretical FLOPs.
- Keep the distinction between **transient memory allocations** and **saved activation memory** explicit.