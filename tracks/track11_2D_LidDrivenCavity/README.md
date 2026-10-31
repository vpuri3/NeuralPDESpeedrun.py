# 11 LidDrivenCavity (2D): Lid-driven cavity, Re = 5000

```text
Steady incompressible Navier-Stokes in the unit square [0, 1]^2, Re = 5000 (nu = 1/Re):

    (u . grad) u + grad p = nu Lap u,      div u = 0
    u = 1 - cosh(50 (x - 0.5)) / cosh(25),  v = 0     on the lid y = 1
    u = v = 0                                        on the other three walls

(The lid profile is 1 except within ~0.05 of the corners, which removes the corner singularities.)
Pressure is not scored. RelL-inf is scored only at points >= 0.05 from the walls.

Output: (u, v) on the 256 x 256 grid x, y = linspace(0, 1, 256); shape (2, 256, 256).
```

**Accuracy bar:** RelL2 ≤ 0.05, RelL∞ ≤ 0.1 (per field) (RelL∞ only at points ≥ 0.05 from the walls). **Budget:** 1000 s of wall-clock solve time.

![reference](figures/reference.png)

## Reference

published data (jaxpi)


## Records

No records yet.
