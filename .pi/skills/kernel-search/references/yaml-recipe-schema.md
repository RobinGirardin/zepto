# YAML recipe spec (Section 8)

Emit a **machine-readable spec block** for future Zepto code. Place inside a fenced `yaml` block in the report.

## Schema

```yaml
region_kind: <e.g. softmax, rmsnorm, gqa, linear_ce>
recommended_region_ids:
  - id: region/<name>
    variant: <liger | hub-xpu | hub-mps | flash2 | …>
    hardware_gate: <any | cuda | rocm | xpu_only | mps_only | exclude_xpu_mps>
    fusion_boundary: <A|B|C|D>
    status: <registered | to_implement>
pattern_rule:  # identity chain this region replaces
  op_families: [Multiply, Exp, ReduceSum, Divide]  # example
recipe:
  forward_flops: "<closed form or per-element * numel>"
  backward_flops: "<closed form>"
  forward_flops_per_element: <int if applicable>
  backward_flops_per_element: <int if applicable>
  materialize_<aux>: <bool>   # e.g. materialize_rstd
  save_<aux>: <bool>          # e.g. save_rstd, save_P
  elided_temps: [<tensor names>]
  saved_backward: [<tensor names + shapes>]
  resource_events_forward: [<ALLOCATE/SAVE sequence>]
  resource_events_backward: [<if distinct>]
  numerics_tags: [<stable_fp32, causal_structural, …>]
capabilities: [<fused>]
priority: <1-7 or optional>
```

## Rules

- If multiple backend variants share the same math but differ in saved-tensor policy (Liger vs MPS mlx-rmsnorm), emit **separate recipe variants** under `recommended_region_ids`.
- Cross-check registered regions: `region/linear`, `region/layernorm`, `region/relu`, `region/rmsnorm`, `region/xielu`.
- `status: to_implement` → add a backlog row in §9.

## Resource event kinds

Use exact kinds: `ALLOCATE`, `RELEASE`, `SAVE`, `PERSIST`, `ALIAS`, `WORKSPACE`.

Example chains:

```
Identity lowering (unfused):
  ALLOCATE(scores) → … → ALLOCATE(exp_scores) → ALLOCATE(P) → SAVE(P)

Fused region leaf:
  ALLOCATE(y) → ALLOCATE(rstd) → SAVE(rstd)   # RMSNorm Liger
  ALLOCATE(P) → SAVE(P)                       # fused softmax boundary A
  ALLOCATE(out)                               # Flash: no (h,S,S) temps
```
