# Golden FLOP / VRAM notes: GatedRMSNorm

Cross-check against `_workspace/research.md` §4–§6 and [`docs/kernel/gated-rms-norm.md`](../../../docs/kernel/gated-rms-norm.md).

## Fused leaf (kernel-accurate)

Let \(n = \lvert \mathbf y \rvert = \lvert \mathbf z \rvert\).

\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 10n, \qquad
\mathrm{FLOPs}_{\mathrm{bwd}} =
\begin{cases}
14n & \text{training} \\
0 & \text{inference-only}
\end{cases}
\]

## Identity lowering (not the default Zepto leaf)

\[
\mathrm{FLOPs}_{\mathrm{identity,fwd}} = 10n + 2 S_{\mathrm{row}},
\quad S_{\mathrm{row}} = n / d.
\]

Do not stack `region/rmsnorm` + `region/silu` on the same `GatedRMSNorm` provenance — double-count risk (research §3).

## Qwen3-Next GDN output norm spot check

\(S=8192\), \(H_v=32\), \(d=128\), bf16:

| Quantity | Value |
|----------|-------|
| \(n = S H_v d\) | \(33{,}554{,}432\) |
| Fused forward FLOPs | \(10n \approx 3.36 \times 10^8\) |
| Saved `rstd` bytes (fp32) | \(\approx S H_v \times 4 \approx 1\,\mathrm{MiB}\) |
