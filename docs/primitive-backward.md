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
