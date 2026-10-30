# track8: Stokes lid-driven cavity

```text
Lid-driven cavity on the square [-1, 1]^2.
Steady incompressible Stokes flow:

    - Lap u + grad p = 0,      div u = 0

Boundary conditions (Dirichlet velocity):
    u = 1, v = 0     on the lid y = 1 (including the corners (+-1, 1))
    u = v = 0        on the other three walls

The velocity is discontinuous at the two top corners. RelL2 is scored over all output points; L-inf
only over points at distance >= 0.05 from the walls (problem.linf_margin). Pressure is NOT scored.

Output: (u, v) on the uniform 129 x 129 grid x, y = linspace(-1, 1, 129);
shape (2, 129, 129).
```

![reference](figures/reference.png)

## Reference

single-element Legendre PN-PN-2 (spec), N=512, Uzawa CG

- `check_N384_vs_N512_linf_scored`: 4.987e-05
- `check_N384_vs_N512_linf_all`: 4.445e-03
- `check_N384_vs_N512_rel_l2`: 2.075e-04

## Records

Current record: **0.1148 s** (record #0: classical baseline, maintainers, 2026-10-30, NVIDIA GeForce RTX 2080 Ti).

| date | author | name | kind | time [s] | RelL2 | L∞ | hardware |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | maintainers | record #0: classical baseline | classical | 0.1148 | 3.14e-03 | 3.69e-03 | NVIDIA GeForce RTX 2080 Ti |

![activity](figures/activity.png)
