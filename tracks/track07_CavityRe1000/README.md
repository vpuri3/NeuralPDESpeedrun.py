# track07_CavityRe1000: Lid-driven cavity, Re = 1000, impulsive start, t in [0, 5]

```text
Lid-driven cavity on the square [-1, 1]^2.
Unsteady incompressible Navier-Stokes, Re = 1000 (nu = 1/Re), t in [0, 5]:

    u_t + (u . grad) u + grad p = nu Lap u,      div u = 0
    u(0, x, y) = v(0, x, y) = 0      (impulsively started lid)

Boundary conditions (Dirichlet velocity):
    u = 1, v = 0     on the lid y = 1 (including the corners (+-1, 1))
    u = v = 0        on the other three walls

The velocity is discontinuous at the two top corners. RelL2 is scored over all output points; L-inf
only over points at distance >= 0.05 from the walls (problem.linf_margin). Pressure is NOT scored.

Output: (u, v) on the uniform 129 x 129 grid x, y = linspace(-1, 1, 129) at the 10
times t = 0.5, 1.0, ..., 5; shape (2, 10, 129, 129).
```

**Accuracy bar:** RelL2 ≤ 0.001, L∞ ≤ 0.001 (L∞ only at points ≥ 0.05 from the walls).


## Records

No records yet.
