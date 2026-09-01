This document describes the backward pass
behavior of all the implemented primitives.

Note that $*$ corresponds to a elementwise multiplication, and $\cdot$ to dot-product or matrix multiplication.
This distinction is important because both require diffent FLOP computation formulas.

* *Elementwise multiplication*: matrices have the same shape. As such, the number of elements in one of the matrices corresponds to the number of operation is requires.
* *matrix multiplication*: requires to use the *2mnk* rule.

## Broadcasting

All elementwise binaries and `MatMul` support NumPy-style broadcasting in the forward pass.
Backward behavior follows these rules:

**Gradient-shape invariant.** For every port in `gradient_outputs`, the produced gradient
has the same shape as the forward tensor bound to that port.

**Sum, not average.** Broadcast backward reduces by **sum** (chain rule over reused positions).

**Reduction FLOPs.** When an operand was broadcast, count
`numel(output) - numel(operand)` additions to sum the unreduced gradient back to the
operand shape. `MatMul` applies the same formula over batch prefix axes only:
`numel(unreduced_batch_grad) - numel(operand)`.

**Default unfused memory.** Backward gradient storage is modeled as structural auxiliary
ports (`grad_left`, `grad_right`, and optional `*_unreduced` temporaries). Unreduced VJP
temps follow `ALLOCATE → … → RELEASE`; reduced gradients follow `ALLOCATE → PERSIST`.
`Add` / `Subtract` pass the upstream gradient through directly and never materialize an
unreduced temp.

**MatMul batch broadcast.** Leading dimensions (`shape[:-2]`) broadcast independently of
the contracting matrix block. A rank-2 weight `(K, N)` pairs with activations
`(B, S, K)` as batch `(B,)` broadcast against `()`. Folded-GEMM lowering may omit the
unreduced batch temp; the structural default charges the unfused cost.

## Multiply

For $Y = A * B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} * B
$$

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y} * A
$$

> When $B$ was broadcast, materialize an output-shaped VJP temporary, then sum-reduce to
> $B$'s shape. When both operands share the output shape, persist reduced gradients only.

## Divide

For $Y = A / B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} * \frac{1}{B}
$$

$$
\frac{\partial L}{\partial B}
= \frac{\partial L}{\partial Y} * -AB^{-2}
= \frac{\partial L}{\partial Y} * -\frac{A}{B^2}
$$

## MatMul

For $Y = A \cdot B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} \cdot B^T
$$

$$
\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial Y}
$$

> Leading batch dimensions broadcast NumPy-style; they no longer must match between
> operands. When a weight's batch prefix was broadcast (e.g. activations `(32, 128, 512)`
> times weight `(512, 64)`), the unfused backward forms a `(32, 512, 64)` temporary and
> sums to `(512, 64)`.

> Note: order matters here — matrix multiplication doesn't commute. $A$ is $(m \times n)$, $B$ is $(n \times p)$, so $\partial L/\partial Y$ is $(m \times p)$. Only $\partial L/\partial Y \cdot B^T$ (shape $m \times n$) and $A^T \cdot \partial L/\partial Y$ (shape $n \times p$) are valid — the reverse orderings don't even have compatible shapes in general.

## Add

For $Y = A + B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y}
$$

> Note: if $A$ or $B$ was broadcast to a larger shape during the forward pass, the corresponding gradient must be summed over the broadcast dimensions to get back to the original shape.
>
> **Why:** broadcasting implicitly reuses the same value at multiple positions in the output. For example, if $A$ has shape $(3, 4)$ and $B$ has shape $(4,)$ (a single row broadcast down to all 3 rows), then $Y = A + B$ has shape $(3, 4)$, and each element $B_j$ was added into **all 3 rows** of $Y$:
>
> $$Y_{i,j} = A_{i,j} + B_j \quad \text{for } i = 0, 1, 2$$
>
> Since $B_j$ contributed to three different outputs, its gradient is the **sum** of the upstream gradient over every position it was reused — this is just the multivariable chain rule (multiple paths → sum the contributions), the same idea used for LayerNorm's $\mu$ and $\sigma^2$:
>
> $$\frac{\partial L}{\partial B_j} = \sum_{i=0}^{2} \frac{\partial L}{\partial Y_{i,j}}$$
>
> So in code, $\partial L/\partial Y$ (shape $(3,4)$) gets summed over axis 0 to produce a gradient of shape $(4,)$ matching $B$. More generally: sum over every axis where $B$'s original shape was $1$ (or absent) but the broadcast output shape was larger. $A$ here wasn't broadcast, so its gradient needs no summation and stays shape $(3, 4)$.

## Subtract

For $Y = A - B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial B} = -\frac{\partial L}{\partial Y}
$$

> Same broadcasting note as Add applies to $B$'s gradient before the sign flip.

## Sqrt

For $Y = \sqrt{A}$

$$
\frac{\partial L}{\partial A} =
\frac{\partial L}{\partial Y} * \frac{1}{2\sqrt{A}} =
\frac{\partial L}{\partial Y} * \frac{1}{2Y}
$$

> Note: reusing the forward output $Y$ avoids recomputing the square root in the backward pass. Watch for numerical instability as $A \to 0$.

## Transpose

For $Y = A^T$

$$
\frac{\partial L}{\partial A} = \left(\frac{\partial L}{\partial Y}\right)^T
$$

## Reshape

For $Y = \text{reshape}(A, \text{shape})$

$$
\frac{\partial L}{\partial A} = \text{reshape}\left(\frac{\partial L}{\partial Y}, \ \text{shape}_A\right)
$$

> Note: reshape doesn't move or change any values, so the backward pass just reshapes the incoming gradient back to $A$'s original shape.

## Identity

For $Y = A$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y}
$$

## Cast

> Forward: $y = \mathrm{cast}(x, T)$ with fixed target dtype $T$; shape unchanged.
>
> Backward: $\partial L/\partial x = \mathrm{cast}(\partial L/\partial y,\, \dtype(x))$.
>
> **Broadcast.** None.
>
> **Saved.** None — input and target dtypes are fixed by the operation declaration.
>
> **FLOPs.** 0 forward and backward (dtype conversion is a memory/metadata op).
>
> **Memory.** Forward allocates output at `itemsize(T) × numel(x)`. Widening
> (e.g. bf16→fp32) doubles bytes; narrowing halves them. Backward allocates
> `grad_input` at the input dtype when `input.requires_grad`.

## Split

For $Y_1, Y_2 = \text{split}(A)$

$$
\frac{\partial L}{\partial A} = \text{concat}\left(\frac{\partial L}{\partial Y_1}, \ \frac{\partial L}{\partial Y_2}\right)
$$

> Note: split is the forward-mode inverse of concat, so its backward is concat — the gradients w.r.t. each output chunk are stitched back together along the split axis.

## Max

For $Y = \max(A, B)$ (elementwise)

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} \cdot \mathbb{1}[A > B]
$$

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y} \cdot \mathbb{1}[B > A]
$$

In practice, however the binary masks are not recomputed, but stored. This is much more efficient, are it allows the storageof a single binary mask (the other one is bitwise NOT), intead of 2 full matrices.

> Broadcast operands follow the same unreduced-temp + sum-reduction pattern as Multiply.

> Note: the gradient routes entirely to whichever input was larger at that position; the other input gets zero. Ties ($A = B$) are a convention choice (commonly split or routed to one side).

## Min

For $Y = \min(A, B)$ (elementwise)

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} \cdot \mathbb{1}[A < B]
$$

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial Y} \cdot \mathbb{1}[B < A]
$$

In practice, however the binary masks are not recomputed, but stored. This is much more efficient, are it allows the storageof a single binary mask (the other one is bitwise NOT), intead of 2 full matrices.

> Broadcast operands follow the same unreduced-temp + sum-reduction pattern as Multiply.

> Note: mirror of Max — gradient routes to whichever input was smaller.

## ReduceSum

**Forward.** Add every entry along the chosen axes. Example: a `(3, 4)` table
summed along rows becomes a length-4 vector (one total per column). If we
keep the empty axis (`keepdim=True`) that result stays `(1, 4)` instead of
`(4,)`. Axis `-1` means the last axis. Listing the same axis twice is ignored. Summing no axes leaves
the tensor unchanged. A single number with no axes is rejected. Summing every
axis yields one number (shape `()`).

$$
Y = \mathrm{sum}(A, \mathrm{axes})
$$

**Backward.** Each input position went into exactly one output sum, so the
input's gradient is the output's gradient **copied** to every position that
was added together. Stretching a shorter tensor is not arithmetic, so backward
costs 0 FLOPs. Forward costs *(input elements − output elements)* additions
(adding *n* numbers takes *n*−1 adds).

$$
\frac{\partial L}{\partial A}
= \mathrm{copy}\!\left(\frac{\partial L}{\partial Y},\ \text{shape of } A\right)
$$

**Broadcast.** This op does not stretch two tensors together. It only sums
inside one tensor.

**Memory.** Keep nothing from the forward pass (which axes to sum is stored on
the op). Store the input's gradient at the input's shape. Do not allocate a
second buffer at the output's shape: the incoming gradient already has that
shape and is only stretched.

## Pow

**Forward.** For every position, raise the base to the exponent. If one input
is smaller, it is stretched by repeating (for example one exponent applied to
a whole tensor). Cost: one power per output element.

$$
Y = A^{B}
$$

**Backward.** Two separate formulas:

- Gradient for the base: incoming gradient × exponent × base^(exponent − 1).
  Needs the **base** and the **exponent**, not the output.
- Gradient for the exponent: incoming gradient × output × log(base). Needs
  the **base** and the **output**.

The log needs a positive base. A power with a negative or fractional exponent
is undefined at zero (same kind of caveat as square root).

Each formula costs 3 operations per output element, plus extra additions if
that input was stretched and must be summed back to its original shape.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * B * A^{B-1}
\qquad
\frac{\partial L}{\partial B}
= \frac{\partial L}{\partial Y} * Y * \log A
$$

**Broadcast.** Same stretching rules as `Multiply`. If an input was stretched
in the forward pass, backward **adds** those extra copies together so the
gradient matches the original input shape.

**Memory.** Like `Multiply`: if an input was stretched, first compute a
full-size gradient, then sum it down. That full-size temporary exists only for
a stretched input. Save `base` and `exponent` if the base needs a gradient.
Save `base` and `output` if the exponent needs a gradient.

## Exp

**Forward.** Apply \(e^{x}\) to every element. Same shape as the input. We bill
**0 FLOPs**: Zepto's cost model follows PyTorch's `FlopCounterMode` convention,
which does not count transcendental ops (`exp`, `log`, `sin`, `cos`) — same
treatment as `Sin`/`Cos`.

$$
Y = \exp(A)
$$

**Backward.** Multiply the incoming gradient by the **output** (which is already
\(e^{x}\)), instead of computing exp again. We still bill **0 FLOPs** for both
forward and backward (same free-transcendental convention as `Sin`/`Cos`). The
formula itself does not change — the output is still worth saving, to avoid
recomputing `exp`.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * Y
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the output if the input needs a gradient. Store the input's
gradient at the input's shape. No extra temporary.

## Log

**Forward.** Apply \(\log x\) to every element. Same shape as the input. The
input should be positive. We bill **0 FLOPs** — same free-transcendental
convention as `Exp`/`Sin`/`Cos` (PyTorch's `FlopCounterMode` doesn't count
these either).

$$
Y = \log(A)
$$

**Backward.** Divide the incoming gradient by the **input** \(x\) (not by the
output). We still bill **0 FLOPs** for both forward and backward (same
convention as `Exp`).

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \frac{1}{A}
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

## Sin

**Forward.** Apply sine to every element. Same shape as the input.

$$
Y = \sin(A)
$$

**Backward.** Multiply the incoming gradient by \(\cos(x)\). That needs the
**input**, not the output (you cannot recover the sign of cosine from sine
alone). We still bill **0 FLOPs** for both forward and backward (trig is
treated as free in this cost model). The formula itself does not change.

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \cos(A)
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

## Cos

**Forward.** Apply cosine to every element. Same shape as the input.

$$
Y = \cos(A)
$$

**Backward.** Multiply the incoming gradient by \(-\sin(x)\). That needs the
**input**. Forward and backward are billed at **0 FLOPs** (same free-trig
convention as `Sin`).

$$
\frac{\partial L}{\partial A}
= \frac{\partial L}{\partial Y} * \bigl(-\sin(A)\bigr)
$$

**Broadcast.** None. Input and output have the same shape.

**Memory.** Keep the input if it needs a gradient. Store the input's gradient
at the input's shape. No extra temporary.

## Concat

**Forward.** Stack at least two tensors along one axis, like placing strips
side by side. They must have the same number of dimensions, and every axis
*except* the join axis must have the same size. Along the join axis, the
output length is the sum of the input lengths. A single number with no axes is
rejected. No arithmetic: 0 FLOPs.

$$
Y = \mathrm{concat}(A_0,\ldots,A_{n-1},\ \mathrm{axis})
\quad (n \ge 2)
$$

**Backward.** Cut the output's gradient back into the same-sized pieces and
hand each piece to the matching input. No arithmetic: 0 FLOPs. Today's `Split`
only cuts axis 0; concat on another axis is still valid, but is not the reverse
of that `Split` until `Split` also takes an axis.

$$
\frac{\partial L}{\partial A_k}
= \text{the }k\text{-th slice of }\frac{\partial L}{\partial Y}
\text{ along the join axis}
$$

**Broadcast.** None. Other axes must match exactly; a size-1 axis is not
stretched to match a larger one.

**Memory.** Keep nothing (the piece sizes are already known from the input
shapes). Store one gradient per input that needs one, at that input's shape.
Inputs that do not need a gradient still count toward the cut positions, so
the remaining pieces line up.

## Gather

**Forward.** Along a chosen axis, replace that axis with the index tensor's
shape, and at each output position copy the input slice whose number is in
`index`. Example: input rows `(5, 8)`, indices `[0, 4, 0]` along axis 0 →
output shape `(3, 8)` (row 0, row 4, row 0 again). Indices are integers in
range. This is ordinary “pick these rows,” not the other gather that requires
the index to have as many dimensions as the table. A table stored as
`(d, vocab)` would need a different axis plus a later transpose; that is the
Module's job, not this op. Cost: 0 FLOPs.

**Backward.** Send each output gradient back to the row it was copied from. If
the same row was picked more than once, **add** those gradients together. The
index list has no gradient (they are discrete choices). The input values
themselves are not needed — only the indices. Cost: 0 FLOPs (collision adds
are not billed).

$$
\frac{\partial L}{\partial A}
= \text{put each } \tfrac{\partial L}{\partial Y} \text{ back at } I
\text{ and add duplicates}
$$

**Broadcast.** The index is not stretched against the input. It *replaces* the
chosen axis: output shape is “input, with that axis swapped for the index
shape.”

**Memory.** Keep `index` if the input needs a gradient. Store the input's
gradient at the input's shape (start from zeros, then add each piece back).
Do not treat this as a view of the output's gradient: even when the two
tensors have the same number of elements, repeats like `[0, 0, 0]` still add
into one row. No extra full-size temporary.

## RepeatKV

**Forward.** Along the chosen axis, copy each slice `n_rep` times in a row
(`n_rep ≥ 1`). Example: two heads and `n_rep = 3` become six heads
`[h0, h0, h0, h1, h1, h1]`. Cost: 0 FLOPs (copies only). When `n_rep = 1`,
the output is the same memory as the input.

**Backward.** Each original slice was reused `n_rep` times, so its gradient is
the **sum** of those copies (not the average). Cost: *(output elements −
input elements)* additions when `n_rep > 1` and the input needs a gradient;
otherwise 0. Keep nothing (`n_rep` and `axis` are stored on the op).

$$
\frac{\partial L}{\partial A_{i}}
= \sum_{j=0}^{n_{\mathrm{rep}}-1}
\frac{\partial L}{\partial Y_{i \cdot n_{\mathrm{rep}} + j}}
$$

**Broadcast.** This copies one chosen axis on purpose. It does not stretch
size-1 axes the way `Add` does. A grouped-query Module must name the **head**
axis (`axis=1` on a `(batch, n_kv, seq, head_dim)` tensor). Repeating axis 0
of that layout would copy the **batch**, which is the wrong thing.

**Memory.** Store the input's gradient at the input's shape. Do not allocate a
second full-size buffer: the incoming gradient already has the output shape
and is only added together. When `n_rep = 1`, reuse the input's storage
(like `Identity`) and do not allocate a separate input gradient — the output's
gradient *is* the input's gradient.

## MaterializedCausalMask

**Forward.** Build one table \(M\) of shape `(1, S, S)` where \(S\) is the
sequence length. For query row \(i\) and key column \(j\): 0 if \(j \le i\)
(allowed), \(-\infty\) if \(j > i\) (blocked). No inputs, no arithmetic
(0 FLOPs). The table is a constant: it is not trained.

$$
M_{0,i,j} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

**Backward.** None. There is nothing to differentiate. Adding this table to
attention scores is a later `Add`.

**Broadcast.** This op does not stretch anything. A later `Add` stretches
`(1, S, S)` across batch and heads when scores are `(batch, heads, S, S)`.

**Memory.** Allocate the table and keep it for the whole run. Keep nothing
else.

## Where

**Forward.** At each position: if the condition is true, take the first value,
otherwise the second. Stretch **all three** inputs to the same shape first
(size-1 axes grow; extra leading axes on the condition also count). Example:
condition `(3, 4)`, true values `(3, 1)`, false values `(1,)` → output
`(3, 4)`, not `(3, 1)`. Cost: 0 FLOPs (picks, no arithmetic). The output needs
a gradient if either *value* does — not if only the condition does.

$$
Y =
\begin{cases}
T & C \text{ is true} \\
F & C \text{ is false}
\end{cases}
$$

**Backward.** The condition is a discrete choice, so it gets no gradient.
Where the condition was true, the incoming gradient goes to the true-value
input (the false side gets 0 there), and the other way around. Stretch the
condition to the output shape before applying that mask. If a value input was
stretched in the forward pass, **add** those extra copies back so the gradient
matches its original shape. Zeros on the unused side still take part in that
sum. Cost: one multiply per output element for each value that needs a
gradient, plus the usual extra additions if that value was stretched.

$$
\frac{\partial L}{\partial T}
= \mathrm{sum\_to\_shape}\!\left(
  \frac{\partial L}{\partial Y} * \mathbb{1}[C],\
  \text{shape of } T
\right)
\qquad
\frac{\partial L}{\partial F}
= \mathrm{sum\_to\_shape}\!\left(
  \frac{\partial L}{\partial Y} * \mathbb{1}[\neg C],\
  \text{shape of } F
\right)
$$

**Broadcast.** Stretch condition, true values, and false values together —
never infer the output shape from the two values alone.

**Memory.** Like `Multiply`: if a value was stretched, first compute a
full-size masked gradient, then sum it down. That temporary exists only for a
stretched value. Keep the condition if either value needs a gradient. Do not
produce a gradient for the condition.
