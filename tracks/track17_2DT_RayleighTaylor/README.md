# 17 RayleighTaylor (2D + Time): Rayleigh-Taylor instability, Ra = 1e6

```text
Rayleigh-Taylor instability (Boussinesq), Ra = 1e6, Pr = 0.71, on x in [0, 1], y in [0, 2], t in [0, 8.5]:

    u_t + (u . grad) u + p_x = a1 Lap u
    v_t + (u . grad) v + p_y = a1 Lap v + a2 T
    u_x + v_y = 0
    T_t + (u . grad) T       = a4 Lap T,        a1 = sqrt(Pr / Ra), a2 = 1, a4 = 1 / sqrt(Ra Pr)

Boundary conditions as in the benchmark formulation: u = v = 0 and T = 0 on the walls y = 0 and y = 2; x is
treated as periodic. (The reference data are consistent with symmetric side walls at x = 0, 1.)
Initial condition: (u, v, p, T) at t = 0 given as samples on the output grid (problem.fns["ic"]).
The grid is non-uniform (clustered near the edges).

Output: (u, v, T) at the 34 snapshot times problem.axes["t"] (spacing 0.25) on the 100 x 200 (x, y) grid
problem.axes["x"], ["y"]; shape (3, 34, 100, 200). Pressure is not scored.
```

**Accuracy bar:** RelL2 ≤ 0.005, RelL∞ ≤ 0.01. **Budget:** 1000 s of wall-clock solve time.

![reference](figures/reference.png)
![reference](figures/reference.gif)

## Reference

published data (jaxpi)


## Records

Current record: **396.8 s** (PirateNet + SOAP, maintainers, 2026-10-31, NVIDIA H100 80GB HBM3 (JAX)).

| date | author | name | kind | time [s] | RelL2 | RelL∞ | passes | hardware |
|---|---|---|---|---|---|---|---|---|
| 2026-10-31 | maintainers | PirateNet + SOAP | neural | 396.8 | 8.63e-01 | 9.98e-01 | no | NVIDIA H100 80GB HBM3 (JAX) |

![accuracy_vs_time](figures/accuracy_vs_time.png)
