# Romberg Integration

*A numerical method for approximating definite integrals*

## Overview

Romberg integration is a numerical technique for approximating definite integrals that combines the **trapezoidal rule** with **Richardson extrapolation**. It produces highly accurate results from relatively few function evaluations, especially for smooth, well-behaved functions.

## The Core Idea

The trapezoidal rule alone converges fairly slowly: its error shrinks proportionally to $h^2$, where $h$ is the step size. Romberg integration speeds this up by computing several trapezoidal estimates at different step sizes, then combining them in a way that cancels out the leading error terms, leaving an error that shrinks much faster (roughly $h^4$, $h^6$, $h^8$, and so on).

## Step 1: Build Trapezoidal Estimates

For an integral of $f(x)$ from $a$ to $b$, compute the composite trapezoidal rule using successively halved step sizes:

$$T_{i,0} = \text{trapezoidal estimate using } n = 2^i \text{ subintervals}, \quad i = 0, 1, 2, \dots$$

Each new estimate reuses function evaluations from the previous level, so doubling the resolution doesn't waste any prior work.

## Step 2: Richardson Extrapolation

The trapezoidal rule's error can be written as a series in even powers of $h$:

$$T(h) = I + c_1 h^2 + c_2 h^4 + c_3 h^6 + \cdots$$

where $I$ is the true integral value. Given two estimates at step sizes $h$ and $h/2$, the $h^2$ term can be eliminated algebraically:

$$T_{i,1} = \frac{4T_{i,0} - T_{i-1,0}}{3}$$

This produces a new estimate with error $O(h^4)$, instead of $O(h^2)$. Repeating the process (combining successive columns) eliminates progressively higher-order error terms using the general recursive formula:

$$T_{i,k} = \frac{4^k T_{i,k-1} - T_{i-1,k-1}}{4^k - 1}$$

## The Romberg Table

Results are arranged in a triangular table. The first column holds the raw trapezoidal estimates; each subsequent column is computed from the one before it. The bottom-right entry is typically the most accurate estimate of the integral.

| i | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| 0 | $T_{0,0}$ | | | |
| 1 | $T_{1,0}$ | $T_{1,1}$ | | |
| 2 | $T_{2,0}$ | $T_{2,1}$ | $T_{2,2}$ | |
| 3 | $T_{3,0}$ | $T_{3,1}$ | $T_{3,2}$ | $T_{3,3}$ |

## Worked Example

Integrate $f(x) = \dfrac{1}{1+x^2}$ over $[0, 1]$. The exact value is $\arctan(1) - \arctan(0) = \pi/4 \approx 0.7853981634$.

First, the raw trapezoidal estimates:

- $n = 1$ (h = 1): $T_{0,0} = 0.750000$
- $n = 2$ (h = 0.5): $T_{1,0} = 0.775000$
- $n = 4$ (h = 0.25): $T_{2,0} = 0.782794$
- $n = 8$ (h = 0.125): $T_{3,0} = 0.784747$

Applying Richardson extrapolation column by column produces the full table:

| i | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| 0 | 0.750000 | | | |
| 1 | 0.775000 | 0.783333 | | |
| 2 | 0.782794 | 0.785392 | 0.785529 | |
| 3 | 0.784747 | 0.785398 | 0.785399 | 0.785397 |

**True value: π/4 ≈ 0.7853981634**

Notice that column $k = 0$ (raw trapezoidal) is still only accurate to about two decimal places after 8 intervals. By column $k = 1$, the result is already accurate to 5–6 decimal places, using the *same* function evaluations. By columns 2–3, the estimate is essentially at the limit of this example's numerical precision.

## In General

Romberg integration does not simply refine a single trapezoidal estimate. Instead, it:

1. **Generates a ladder of trapezoidal estimates**: T(n=1), T(n=2), T(n=4), T(n=8), ... each one finer than the last.
2. **Combines adjacent estimates** to cancel the leading error term, the Richardson extrapolation step.
3. **Repeats the combining** on each new column to cancel the next error term, and so on.

The trapezoidal rule supplies raw materials of varying quality; Romberg integration is the algoriithm for combining them so the errors cancel out far faster than brute-force refinement of a single estimate ever could.

## Limitations

- Romberg assumes the integrand is smooth: the error expansion in even powers of $h$ relies on this. Functions with discontinuities, kinks, or singularities can converge slowly or give misleading results.
- Works on a single interval: highly oscillatory or sharply localized functions may need adaptive methods or domain splitting first.
