# Golden FLOP / VRAM notes: gated delta scan

Cross-check against `_workspace/research.md` §4–§6 and Atto gated DeltaNet §Computation.

## Zepto identity grouping (default leaf)

Per head per timestep:

\[
8 d_k d_v + 2 d_v + 1 \quad\text{(forward)}, \qquad
16 d_k d_v + 4 d_v + 4 \quad\text{(backward, training)}.
\]

Over \(S\) steps and \(h\) heads:

\[
\mathrm{FLOPs}_{\mathrm{fwd}} = S h (8 d_k d_v + 2 d_v + 1).
\]

## Qwen-style spot check

\(S=8192\), \(h=48\), \(d_k=d_v=128\):

\[
8192 \times 48 \times (8 \times 128 \times 128 + 2 \times 128 + 1)
\approx 5.2 \times 10^{10}\ \text{forward FLOPs}.
\]

Saved fp32 states (training, all timesteps): \(4 \cdot S h d_k d_v \approx 2.0\ \mathrm{GiB}\) per layer.

## Atto paper grouping (documentation only)

Atto reports \(\approx 9 S h d_k d_v\) forward and \(\approx 10 S h d_k d_v\) backward when \(d_k=d_v\) — ratio \(\approx 8/9\) on the dominant forward term vs Zepto §4.1. Zepto fused leaf bills §4.1, not Atto 9/10.
