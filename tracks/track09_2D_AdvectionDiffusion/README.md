# 09 AdvectionDiffusion (2D): 2D advection-diffusion, Pe ~ 1e3

```text
Steady advection-diffusion on the unit square [0, 1]^2:

    -eps Lap u + b . grad u = f,      b = (1, 1),   eps = 0.001
    u = 0 on the boundary
    f(x, y) = g(x) + g(y),   g(s) = s - (exp((s - 1)/eps) - exp(-1/eps)) / (1 - exp(-1/eps))

The solution has exponential boundary layers of width ~eps along the outflow walls x = 1 and y = 1.

Output: u on the tensor grid x_i = y_i = (1 - cos(pi i / 256)) / 2, i = 0..256
(Chebyshev-Gauss-Lobatto points, clustered at the walls so the boundary layers are sampled);
shape (1, 257, 257).
```

**Accuracy bar:** RelL2 ≤ 0.005, RelL∞ ≤ 0.01. **Budget:** 1000 s of wall-clock solve time.

![reference](figures/reference.png)
![reference_slices](figures/reference_slices.png)

## Reference

closed form u = g(x) g(y)

- `check_chebyshev_N96_linf`: 3.342e-05

## Records

No records yet.
