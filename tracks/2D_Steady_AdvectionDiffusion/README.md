# 2D_Steady_AdvectionDiffusion: 2D steady advection-diffusion, cellular flow, Pe = 64

```text
Steady advection-diffusion of temperature T(x, y) on [0, 1] x [0, 2] by a prescribed cellular flow:

    (1 / Pe) (T_xx + T_yy) - u T_x - v T_y = 0,      Pe = 64
    u = -sin(pi x) cos(pi y),   v = cos(pi x) sin(pi y)      (two counter-rotating cells, divergence-free)
    T(0, y) = 0 (cold wall),   T(1, y) = 1 (hot wall),   T_y(x, 0) = T_y(x, 2) = 0 (insulated walls)

The flow sweeps hot fluid from the hot wall along the insulated walls toward the cold wall, and cold fluid
back along the cell boundary y = 1, bending the isotherms into an S around each cell.

Output: T on the uniform grid x_i = i / 256, i = 0..256, y_j = j / 256, j = 0..512;
shape (1, 257, 513).
```

**Accuracy bar:** RelL2 ≤ 0.005, RelL∞ ≤ 0.01. **Budget:** 1000 s of wall-clock solve time.

![reference](figures/reference.png)
![reference_slices](figures/reference_slices.png)

## Reference

Chebyshev collocation on one cell, N = 64, mirrored about y = 1

- `check_N48_linf`: 6.255e-12
- `check_N56_linf`: 6.450e-12
- `check_full_domain_N64x128_linf`: 2.149e-10

## Records

Current record: **251.1 s** (PINN + Gauss-Newton (D-NGD), maintainers, 2026-11-05, NVIDIA H100 80GB HBM3 (JAX)).

| date | author | name | kind | time [s] | RelL2 | RelL∞ | passes | hardware |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | maintainers | Shallow PINN + Levenberg-Marquardt | neural | 1000 | 3.22e-08 | 1.25e-07 | yes | NVIDIA H100 80GB HBM3 (JAX) |
| 2026-11-05 | maintainers | PirateNet + SOAP | neural | 352.6 | 2.46e-03 | 4.52e-03 | yes | NVIDIA H100 80GB HBM3 (JAX) |
| 2026-11-05 | maintainers | PINN + Gauss-Newton (D-NGD) | neural | 251.1 | 5.03e-07 | 1.61e-06 | yes | NVIDIA H100 80GB HBM3 (JAX) |

![accuracy_vs_time](figures/accuracy_vs_time.png)

![activity](figures/activity.png)
