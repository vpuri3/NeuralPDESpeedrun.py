# 07 AdvectionDiffusion (1D): 1D steady advection-diffusion, boundary layer, eps = 0.01

```text
Steady advection-diffusion on [0, 1] (flow toward x = 0):

    -eps u'' - u' = 1,      eps = 0.01
    u(0) = u(1) = 0

Away from x = 0 the solution follows the outer solution 1 - x; it drops to 0 across a boundary layer of width
~eps at the outflow wall x = 0 (u = 1/2 at x ~ eps ln 2, slope u'(0) ~ 1/eps).

Output: u at x_i = i / 1000, i = 0..1000; shape (1, 1001).
```

**Accuracy bar:** RelL2 ≤ 0.005, RelL∞ ≤ 0.01. **Budget:** 1000 s of wall-clock solve time.

![reference](figures/reference.png)

## Reference

exact solution

- `check_chebyshev_N128_linf`: 1.366e-14

## Records

No records yet.
