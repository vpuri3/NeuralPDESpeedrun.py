# 2D_Steady_Helmholtz: 2D Helmholtz, plane wave scattered by a square in a disk, k = 60

```text
Helmholtz's equation for the wave r(x, y) reflected by a square obstacle in a disk (time-harmonic, e^(-i omega t)):

    -Lap r - k^2 r = 0,        k = 60
    domain: the disk |(x, y) - c| < R minus the square |x - c_x| + |y - c_y| <= a,
            c = (0.8, 0.5),  R = 0.45,  a = 0.1 / sqrt(2)   (a square of side 0.1, rotated by 45 degrees)
    on the square:  r = -exp(-i k x)            (the incident wave exp(-i k x), traveling in -x, is reflected:
                                                 the total field vanishes on the obstacle)
    on the circle:  dr/dn - i k r = 0           (nonreflecting boundary condition, n the outward normal)

r is complex. The square's 4 corners are re-entrant (interior angle 270 degrees): grad r ~ rho^(-1/3) there.

Output: Re r and Im r at the nodes of the given mesh, problem.mesh: quadratic triangles of edge length <= 0.02
(as MATLAB's generateMesh(model, Hmax=0.02)); "points" (n, 2), "triangles" (m, 6: corners counterclockwise, then
the midside nodes of edges 12, 23, 31), "obstacle" / "outer": ids of the nodes on the square / on the circle.
Shape (2, n).
```

**Accuracy bar:** RelL2 ≤ 0.005, RelL∞ ≤ 0.01. **Budget:** 1000 s of wall-clock solve time.


## Records

No records yet.
