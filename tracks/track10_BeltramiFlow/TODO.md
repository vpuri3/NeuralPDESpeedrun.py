# track10_BeltramiFlow: 3D unsteady Beltrami flow (TODO, not set up)

Notes for setting this track up later. Setup as in Jnini, Vella & Zeinhofer, *Gauss-Newton Natural Gradient
Descent for Physics-Informed Computational Fluid Dynamics* (arXiv:2402.10680, §4.2), which uses the
Ethier & Steinman (1994) exact solution.

- **Equations:** 3D unsteady incompressible Navier-Stokes, `u_t + (u·∇)u + ∇p = ν Δu`, `∇·u = 0`, Re = 1 (ν = 1).
- **Domain:** (x, y, z) ∈ [−1, 1]³, t ∈ [0, 1]. Not periodic.
- **Exact solution** (a = d = 1; decays like e^(−t), pressure like e^(−2t)):
  ```
  u = −[ e^x sin(y + z) + e^z cos(x + y) ] e^(−t)
  v = −[ e^y sin(z + x) + e^x cos(y + z) ] e^(−t)
  w = −[ e^z sin(x + y) + e^y cos(z + x) ] e^(−t)
  p = −½ [ e^(2x) + e^(2y) + e^(2z)
           + 2 sin(x + y) cos(z + x) e^(y + z)
           + 2 sin(y + z) cos(x + y) e^(z + x)
           + 2 sin(z + x) cos(y + z) e^(x + y) ] e^(−2t)
  ```
  (General form: u = −a[e^(ax) sin(ay + dz) + e^(az) cos(ax + dy)] e^(−ν d² t), etc.; check a, d, ν against the
  paper before fixing the track.)
- **Data a solver may use:** initial velocity at t = 0 and the Dirichlet velocity on all six faces for t ∈ [0, 1]
  (both from the exact solution). The interior is unknown to the solver; the reference is the closed form.
- **Scoring (to decide):** the paper reports relative L2 of u, v, w, p at the final time t = 1 (pressure only up
  to a constant). Here: score (u, v, w) at a few snapshots (e.g. t = 0.25, 0.5, 0.75, 1) on a tensor grid such as
  33³ or 65³; pressure unscored, as on the other tracks.
- **Reference points from the paper** (relative L2 at t = 1, best seed, fp64 on an A100): Adam ~4e-2,
  BFGS ~3e-4, Gauss-Newton natural gradient ~4e-6. Training used 31 × 31 points per face for the boundary and
  initial conditions and 10,000 interior space-time points.
- **Classical baseline:** e.g. a 3D Chebyshev/Legendre spectral or PN-PN-2 solver with time-dependent Dirichlet
  data. The smooth solution makes spectral methods very fast here; the track is mainly a 3D, time-dependent
  benchmark for neural / Gauss-Newton methods.
