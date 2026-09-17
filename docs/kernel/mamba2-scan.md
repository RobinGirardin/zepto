# Zepto kernel research: Mamba-2 selective SSM scan

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone selective state-space scan leaf (`SelectiveSSMScan`); prefill (\(S>1\)) and decode (\(S=1\)) with optional recurrent state port; training + inference; CUDA primary (`mamba-ssm` Triton SSD); identity reference is S-unrolled at compose time
**Context:** Core recurrence in Nemotron 3.5 Lightning **Mamba-2** mixers after depthwise causal conv. Zepto module [`SelectiveSSMScan`](../src/zepto/modules/selective_ssm_scan.py) is registered; fused region `region/mamba2_scan` is a **stub** in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py). Parent `Mamba2Mixer` supplies projected `x`, raw \(\Delta\), and grouped `B`/`C` — **not** billed in this leaf. Input-independent parameters `decay_rate` (\(\exp(A_{\log})\)), `d_skip` (\(D\)), and `dt_bias` **are** owned by the scan module. Default cost leaf should be **kernel-accurate** fused scan (`mamba2_chunk_scan` prefill / `mamba2_selective_state_update` decode), not the sum of per-token graph overhead or compose-time `Split`/`Concat` nodes.

---

## Section 0: Mathematical definition

**Mamba-2 selective SSM** maintains a matrix-valued state \(\mathbf H_t \in \mathbb{R}^{p \times N}\) per head and applies input-dependent discretization each timestep ([Mamba-2, Dao & Gu, 2024](https://arxiv.org/abs/2405.21060); Zepto contract in [step5 §Phase 5](../docs/plans/step5-stateful-mixers.md)):

\[
\begin{aligned}
A_i &= -\exp(A_{\log,i}) < 0,\\
\overline{\Delta}_{t,i} &= \operatorname{softplus}(\Delta_{t,i} + \mathrm{dt\_bias}_i) > 0,\\
\overline{A}_{t,i} &= \exp(\overline{\Delta}_{t,i} A_i),\\
\overline{B}_{t,i,n} &= \overline{\Delta}_{t,i} B_{t,\gamma(i),n},\\
H_{t,i,r,n} &= \overline{A}_{t,i} H_{t-1,i,r,n} + x_{t,i,r}\,\overline{B}_{t,i,n},\\
y_{t,i,r} &= \sum_{n=1}^{N} H_{t,i,r,n} C_{t,\gamma(i),n} + D_i x_{t,i,r},
\end{aligned}
\]

with head index \(i\), channel \(r \in \{1..p\}\), state coordinate \(n\), and B/C sharing group \(\gamma(i) = \lfloor i / (h/G) \rfloor\) for \(G\) groups.

**I/O shapes (Zepto module, batch \(B=1\) in reference graph):**

| Tensor | Shape | Role |
|--------|-------|------|
| `x` | \((S, h, p)\) | SSM input after conv split (per-head channels) |
| `delta` | \((S, h)\) | Raw \(\Delta\) before bias + softplus |
| `b` | \((S, G, N)\) | Selective \(B\) (group-shared) |
| `c` | \((S, G, N)\) | Selective \(C\) (group-shared) |
| `scan_state_in` | \((h, p, N)\) or `None` | Initial state; fp32 accumulation |
| `output` | \((S, h, p)\) | Scan outputs (cast to activation dtype) |
| `scan_state_out` | \((h, p, N)\) | Final state for horizon / decode |

**Learned parameters at scan boundary (not in parent projections):**

| Parameter | Shape | Role |
|-----------|-------|------|
| `decay_rate` | \((h,)\) | \(\exp(A_{\log})\) — positive stored form for Zepto parameter ports |
| `d_skip` | \((h,)\) | Skip \(D_i\) per head |
| `dt_bias` | \((h,)\) | Added to raw \(\Delta\) before softplus |

**Numerics policies:**

- Recurrent state \(\mathbf H\) accumulates in **fp32** (`RMSNORM_COMPUTE_DTYPE`); sequence output cast back to bf16/fp16 ([`selective_ssm_scan.py`](../src/zepto/modules/selective_ssm_scan.py)).
- \(\overline{A}_{t,i} = \exp(\overline{\Delta}_{t,i} A_i)\) with \(A_i < 0\) and \(\overline{\Delta} > 0\) gives \(0 < \overline{A} \le 1\) (stable decay).
- Nemotron does **not** pass Mamba gate `z` into the scan; gating is **`GatedGroupedRMSNorm`** downstream ([`mamba2_mixer.py`](../src/zepto/modules/mamba2_mixer.py)).
- B/C group expansion \((S,G,N) \to (S,h,N)\) is layout/broadcast in fused kernels — not extra recurrence FLOPs at the scan leaf.
- Chunk SSD kernels factorize the **same** causal recurrence as the token loop ([`mamba_ssm/ops/triton/ssd_combined.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py)).

**Structural vs eager:**

- **Identity lowering:** Python `for t in range(S)` emits per-step `Exp`, `Multiply`, `ReduceSum`, `Add` at compose time (graph size \(\Theta(S)\), arithmetic \(\Theta(S h p N)\)).
- **Fused leaf:** Triton SSD chunk scan (prefill) or `selective_state_update` (decode); same asymptotic FLOPs, elides per-step HBM temps.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | [`mamba2_chunk_scan`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) / Hub `mamba_ssm` Triton SSD (`_chunk_scan_fwd`, `_chunk_state_fwd`, `_state_passing_fwd` in [`ssd_combined.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py)); decode [`mamba2_selective_state_update`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py); optional [`mamba2_split_conv1d_scan_combined`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) (conv+scan mega-fusion, boundary **B**) |
| **Identity lowering** | Unfused semantic chain | Zepto `SelectiveSSMScan`: S-unrolled discretization + state update + \(C\) contraction + \(D\) skip ([`selective_ssm_scan.py`](../src/zepto/modules/selective_ssm_scan.py)); horizon recurrent port on `scan_state_out` ([step5 §7.5](../docs/plans/step5-stateful-mixers.md)) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/mamba2_scan` (stub) — closed-form **\(f(S,h,p,N)\)** forward/backward; `region/mamba2_scan/decode` when `S=1` + state port |

**Default estimates use the fused region leaf when `requested_capabilities` includes `fused`, not the identity graph node count.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Mamba SSD chunk scan** | `mamba_ssm` Triton [`ssd_chunk_scan`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py) via `mamba2_chunk_scan` | Yes (full scan) | **C** | CUDA (Triton) | Prefill + training |
| **Selective state update** | `mamba2_selective_state_update` / Triton recurrent step | Yes (single step) | **C** | CUDA | Decode / streaming |
| **Conv+scan combined** | `mamba2_split_conv1d_scan_combined` | Yes (conv+scan+split) | **B** | CUDA + Hub | When HF hub kernel enabled |
| **HF Nemotron H** | [`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | Uses Hub fallbacks above | C / B | CUDA + Hub | Prefill vs decode branches |
| **HF Mamba2 generic** | [`Mamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) | Same kernel hooks | C / B | CUDA + Hub | Both |
| **Zepto module (identity)** | `SelectiveSSMScan` | No (S-unrolled) | C | Any | Both; emits recurrent state port |
| **Zepto region (stub)** | `region/mamba2_scan` in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py) | Cost leaf TBD | C | `fused` gate | `status: to_implement` |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **C** — full selective SSM recurrence (state update + readout + \(D\) skip). Sub-leaf **A** only for isolated discretization primitives (not recommended for default billing).

**Mutually exclusive (same scan slot on matched provenance):**

- `region/mamba2_scan/*` vs identity `SelectiveSSMScan` S-unrolled graph — fused region wins when pattern + `fused` capability match.
- `region/mamba2_scan` (scan only) vs **`region/mamba2_mixer`** block envelope — block region subsumes child scan FLOPs/memory when Tier-B composition matches ([step5 §8.1](../docs/plans/step5-stateful-mixers.md)).
- `region/mamba2_scan` vs **`mamba2_split_conv1d_scan_combined`** on the same layer — mega-kernel replaces separate conv + scan leaves ([`depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md) §3).

**Composable:**

- **Mamba2Mixer:** `Linear → DepthwiseCausalConv1d → SelectiveSSMScan → GatedGroupedRMSNorm → Linear` — scan bills only recurrence + read; projections/conv/norm are separate regions.
- **Horizon decode:** prefill uses full \(S\); decode steps use `region/mamba2_scan/decode` with `scan_state_in` **PERSIST** (related gap **G3** recurrent state bytes across steps).

**Execution constraints:**

- Requires `num_heads % num_groups == 0`; B/C are \((S,G,N)\) at boundary ([`selective_ssm_scan.py`](../src/zepto/modules/selective_ssm_scan.py)).
- Chunk kernels need sufficient \(S\) for SSD tiling; decode uses recurrent update (\(S=1\)).
- `requires_grad=False` → backward FLOPs = 0; inference decode persists **one** final state \((h,p,N)\), not all \(S\) checkpoints unless training.
- Batch dimension: reference graph uses \(B=1\); scale FLOPs/bytes by \(B\) for \((B,S,\ldots)\) layouts.

---

## Section 4: Forward FLOPs — step-by-step derivation

Notation: \(S\) sequence length, \(h\) heads, \(p\) head dim (channels per head), \(N\) state size, one elementwise mul/add = **1 FLOP** (Zepto elementwise convention; no MAC doubling on pure elementwise tiles).

### 4.1 Identity lowering (Zepto `SelectiveSSMScan` per timestep \(t\), per head \(i\))

| Step | Primitive | Tile / shape | FLOPs (per head) | Subtotal |
|------|-----------|--------------|------------------|----------|
| 1 | `Add` + Softplus | scalar \(\Delta_{t,i}\) | 4 (`Add`, `Exp`, `Add`, `Log`) | 4 |
| 2 | `ParameterScale` + `Multiply` + `Exp` | form \(\overline{A}_{t,i}\) | 3 | 3 |
| 3 | `Multiply` | \(\overline{B}_{t,i,n} = \overline{\Delta}\, B_{t,n}\) | \(N\) | \(N\) |
| 4 | `Multiply` | decay \(\overline{A} \odot H_{t-1}\) | \(pN\) | \(pN\) |
| 5 | `Multiply` | update \(x_t \odot \overline{B}\) | \(pN\) | \(pN\) |
| 6 | `Add` | \(H_t = \text{decay} + \text{update}\) | \(pN\) | \(pN\) |
| 7 | `Multiply` + `ReduceSum` | \(C\) contraction | \(2pN\) | \(2pN\) |
| 8 | `ParameterScale` + `Add` | \(D \odot x_t\) skip | \(2p\) | \(2p\) |
| | **Per head per step** | | | \(5pN + N + 2p + 7\) |

Over \(S\) steps and \(h\) heads:

\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{identity}} = S h \bigl(5 p N + N + 2 p + 7\bigr).
\]

When \(pN \gg 1\): \(\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{identity}} \approx 5 S h p N\).

### 4.2 Fused region leaf (kernel-accurate)

SSD chunk / selective-update kernels implement the same recurrence; fused path bills the **same** closed form as §4.1 (dominant term \(5 S h p N\)), not extra compose-time `Reshape`/`Split`/`RepeatKV` nodes.

**Decode (\(S=1\)):**

\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{decode}} = h \bigl(5 p N + N + 2 p + 7\bigr).
\]

### 4.3 Paper-comparable (SSD / block factorization)

The Mamba-2 paper reports SSD via structured matmuls over chunks; **asymptotic** causal scan work remains \(\Theta(S h p N)\) for the selective recurrence ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)). Chunk factorization changes **parallelism and memory**, not the leading FLOP exponent. A coarser paper-style grouping that fuses decay+update+read into **\(4 p N\)** MAC-equivalent per head-step would yield **\(4 S h p N\)** — label **paper-comparable** when citing that grouping; **default Zepto leaf = §4.1** (\(5 S h p N\) dominant).

**Closed form (kernel-accurate default):**

\[
\boxed{\mathrm{FLOPs}_{\mathrm{fwd}} = S h \bigl(5 p N + N + 2 p + 7\bigr)}
\]

**Arithmetic intensity (numeric example — Nemotron Mamba-2 scan leaf, \(S=8192\), \(h=64\), \(p=64\), \(N=128\), bf16 \(e=2\)):**

- \(\mathrm{FLOPs}_{\mathrm{fwd}} \approx 8192 \times 64 \times (5 \times 64 \times 128 + 128 + 128 + 7) \approx 2.16 \times 10^{10}\).
- Minimum HBM moved if inputs/outputs touched once (no fusion):  
  \(|x| = S h p e\), \(|\Delta| = S h e\), \(|B|+|C| = 2 S G N e\) with \(G=8\), \(|O| = S h p e\)  
  \(\Rightarrow\) order \((2 h p + h + 2 G N/h \cdot h) S e \approx (2hp + h + 2GN) S e \approx 1.1 \times 10^8\) bytes \(\approx 105\) MiB.
- Intensity \(\approx 2.16 \times 10^{10} / 1.1 \times 10^8 \approx 2.0 \times 10^2\) FLOP/byte (memory-friendlier than full GQA at same \(S\), but not purely bandwidth-bound).

**Apertus-8B cross-check (\(S=8192\), \(d=4096\), \(h=32\), \(d_h=128\)):** Default Apertus stack uses GQA attention, not Mamba-2. If a hypothetical Mamba layer reused \(h=32\), \(p=128\), \(N=128\):  
\(\mathrm{FLOPs}_{\mathrm{fwd}} \approx 8192 \times 32 \times 5 \times 128 \times 128 \approx 2.1 \times 10^{10}\) (same order as Nemotron scan slice at comparable \(h p N\)).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training requires backward-through-time (BPTT) across \(S\) steps. Dominant forward ops (decay, rank-one update along \(N\), \(C\) contraction, \(D\) skip) each admit reverse passes of similar elementwise order.

### 5.1 Saved vs recomputed

| Policy | Saved tensors | Backward effect |
|--------|---------------|-----------------|
| **Identity / generic BPTT** | All \(\mathbf H_t\) for \(t=1..S\) | Reverse recurrence; memory \(\Theta(S h p N)\) |
| **Fused SSD chunk bwd** | Chunk checkpoints / recomputation ([`_chunk_scan_bwd_*`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py)) | Same asymptotic FLOPs; lower peak via chunk policy |
| **Inference** | Final state only | `requires_grad=False` → **0** backward FLOPs |

### 5.2 Backward step table (per head, per timestep, kernel-accurate order-of-magnitude)

| Step | Reverse of | FLOPs (per head) |
|------|------------|------------------|
| 1 | \(D\) skip + output add | \(2p\) |
| 2 | \(C\) contraction + state adjoint | \(3pN\) |
| 3 | Update + decay branch | \(4pN\) |
| 4 | \(\overline{B}\), \(\overline{A}\), softplus chain | \(2pN + N + 7\) |
| | **Total (order)** | \(\approx 10 p N + 2 p + N\) |

**Closed form (Zepto default leaf, ~2× dominant forward MACs on recurrence + read):**

\[
\boxed{\mathrm{FLOPs}_{\mathrm{bwd}} = S h \bigl(10 p N + 2 p + N + 4\bigr)}
\]

when `requires_grad=True`. Equivalently \(\mathrm{FLOPs}_{\mathrm{bwd}} \approx 2 \times \mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{dominant}}\) for \(pN \gg 1\).

**Decode (\(S=1\)):** replace \(S\) with 1 in the boxed formula.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Assume fp32 state accumulation (\(e_s=4\)) for \(\mathbf H\); activations \(e=2\) (bf16) unless noted. Scan leaf only — `x`, \(\Delta\), `B`, `C` treated as **inputs** from parent conv/split.

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|--------------------|------------------|
| **Identity lowering** | Per-\(t\) temps: `a_bar`, `b_bar`, `decayed`, `update`, `weighted`, `y_t` each \(\mathcal{O}(h p N)\) or \(\mathcal{O}(h p)\) | \(\mathcal{O}(h p N)\) if one step at a time | N/A (worst case) |
| **Fused SSD chunk** | Output, chunk workspace; state in registers / tiles | \(\mathcal{O}(h p N + S h p)\) | Per-step full-rank temps |
| **Decode recurrent** | Single-step workspace + state | \(\mathcal{O}(h p N)\) | Full-sequence temps |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto `SAVE` policy |
|------|---------------|-------|-------|---------------------|
| **BPTT default** | \(\mathbf H_t\) all timesteps | \(S \times h \times p \times N\) | \(4 \cdot S h p N\) (fp32 state) | `SAVE(state_checkpoint)` when training and not chunk-recomputed |
| **Inputs** | `x`, \(\Delta\), `B`, `C` | as §0 | \(\mathcal{O}(S(hp + h + 2GN))\) | Parent regions own unless scan saves for fusion |
| **Inference decode** | Final \(\mathbf H_S\) only | \(h \times p \times N\) | \(4 h p N\) | `PERSIST` on horizon recurrent port (**G3**) |

**Nemotron-scale example:** \(S=8192\), \(h=64\), \(p=64\), \(N=128\) → saved states \(\approx 8192 \times 64 \times 64 \times 128 \times 4 \approx 1.0\) GiB per layer (dominant training memory for scan leaf).

### 6.3 Resource event chains

**Identity lowering (unfused):**

```
ALLOCATE(state_0) → for t in 1..S:
  WORKSPACE(discretization) → ALLOCATE(decayed_t) → ALLOCATE(update_t)
  → SAVE(state_t) [training] → ALLOCATE(out_t)
→ RELEASE(decayed_t...) [if not saved]
→ PERSIST(scan_state_out)
```

**Fused region leaf (prefill):**

```
ALLOCATE(output[S,h,p]) → WORKSPACE(ssd_chunk) → SAVE(state_checkpoint policy)
→ PERSIST(scan_state_out) → RELEASE(workspace)
```

**Fused decode:**

```
LOAD(scan_state_in) → WORKSPACE(selective_update) → ALLOCATE(output[1,h,p]) → PERSIST(scan_state_out)
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Zepto identity | `SelectiveSSMScan` | No | C | \(5 S h p N\) (+ lower order) | \(\approx 10 S h p N\) | Materialize per-step temps (worst case) | (lowering graph) |
| SSD chunk | `mamba2_chunk_scan` | Yes | C | \(5 S h p N\) | \(\approx 10 S h p N\) | Chunk workspace; optional final state | `region/mamba2_scan` |
| Selective update | `mamba2_selective_state_update` | Yes | C | \(5 h p N\) per step | \(\approx 10 h p N\) | Single state | `region/mamba2_scan/decode` |
| Conv+scan combined | `mamba2_split_conv1d_scan_combined` | Yes | B | bundled | bundled | Fused w/ conv | `region/mamba2_mixer` |
| Paper SSD grouping | Mamba-2 paper | N/A | C | \(\approx 4 S h p N\) | (not Zepto default) | Chunk matmul temps | cross-check only |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: mamba2_scan
recommended_region_ids:
  - id: region/mamba2_scan
    variant: mamba_ssd_chunk
    hardware_gate: cuda
    fusion_boundary: C
    status: to_implement
  - id: region/mamba2_scan/decode
    variant: selective_state_update
    hardware_gate: cuda
    fusion_boundary: C
    status: to_implement
  - id: region/mamba2_scan
    variant: reference
    hardware_gate: any
    fusion_boundary: C
    status: to_implement
pattern_rule:
  op_families: [Exp, Log, Add, Multiply, ReduceSum, Cast, Reshape, Split, Concat, RepeatKV]
recipe:
  forward_flops: "S * h * (5 * p * N + N + 2 * p + 7)"
  backward_flops: "S * h * (10 * p * N + 2 * p + N + 4)"
  forward_flops_per_element: null
  backward_flops_per_element: null
  materialize_per_step_temps: false
  save_state_all_timesteps: true
  save_scan_state_out: true
  elided_temps: [a_bar_t, b_bar_t, decayed_t, update_t, weighted_t, softplus_intermediate]
  saved_backward:
    - "state_t: [S, h, p, N] fp32 when training and not SSD-checkpointed"
    - "scan_state_out: [h, p, N] for horizon persist"
  resource_events_forward:
    - "ALLOCATE(output)"
    - "WORKSPACE(ssd_chunk_scan)"
    - "PERSIST(scan_state_out)"
  resource_events_backward:
    - "LOAD(saved state_t or recompute from chunk checkpoint)"
  numerics_tags: [fp32_state_accum, softplus_delta, grouped_bc_broadcast, structural_s_unroll_identity]
capabilities: [fused, recurrent_state_port]
priority: 5
variants:
  prefill:
    trigger: "seq_len > 1 or phase == prefill"
    forward_flops: "S * h * (5 * p * N + N + 2 * p + 7)"
  decode:
    trigger: "seq_len == 1 and scan_state_in present"
    forward_flops: "h * (5 * p * N + N + 2 * p + 7)"
    id: region/mamba2_scan/decode
  mega_fusion_exclusion:
    trigger: "mamba2_split_conv1d_scan_combined matched on same mixer"
    note: "Bill region/mamba2_mixer instead of separate conv + scan"
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Mamba-2 / SSD: [Transformers are SSMs: Structured State Space Duality](https://arxiv.org/abs/2405.21060) (Dao & Gu, 2024)
- Original Mamba selective scan: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Mamba-SSM Triton SSD implementation: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py
- HF Mamba2 kernel hooks: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py
- HF Nemotron H Mamba2 mixer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py
- Zepto selective scan module: https://github.com/RobinGirardin/zepto/blob/main/src/zepto/modules/selective_ssm_scan.py (local: `src/zepto/modules/selective_ssm_scan.py`)
- Zepto step5 mixer plan (equations §Phase 5): `docs/plans/step5-stateful-mixers.md`
- Related conv leaf (mega-fusion mutual exclusion): `docs/kernel/depthwise-causal-conv1d.md`

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 5 | `region/mamba2_scan` | §8 | C | Nemotron Mamba-2 prefill; stub in `mixer_stub.py`; recipe from this research |
| 5 | `region/mamba2_scan/decode` | §8 variants.decode | C | Horizon decode + `scan_state_in` port |
| 6 | `region/mamba2_mixer` | §3 mega_fusion | B | Optional `mamba2_split_conv1d_scan_combined` supersedes conv+scan leaves |

**Open gaps (TBD for architect):**

- Exact SSD backward saved-tensor policy vs `save_state_all_timesteps` (match `mamba_ssm` checkpointing when implementing).
- Tier-B `region/mamba2_mixer` subsumption rules vs Tier-A leaf stack (step5 §8.1).
- Atto golden page for Mamba-2 (planned `tests/lowering/regions/derivations/mamba2_scan.md`) — cross-check FLOPs when authored.
