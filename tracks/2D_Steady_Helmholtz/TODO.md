# 2D_Steady_Helmholtz: TODO

The MATLAB PDE Toolbox example "Helmholtz's Equation on Disk with Square Hole"
(https://www.mathworks.com/help/pde/ug/helmholtzs-equation-on-disk-with-a-square-hole.html; geometry from its
`scatterg.m`, boundary conditions from `scatterb.m`): `-Δr - k²r = 0`, k = 60, on the disk of radius 0.45 centered
at (0.8, 0.5) minus a square of side 0.1 rotated by 45° with the same center; `r = -exp(-ikx)` on the square,
`∂r/∂n - ikr = 0` on the circle. The square's re-entrant corners make `∇r ~ ρ^(-1/3)` there.

Done:
- `problem.py`: the problem; the output is Re r and Im r at the nodes of the track mesh (`Problem.mesh`).
- `make_mesh.py` → `data/mesh.npz`: 9528 nodes, 18758 linear triangles (Triangle, edge ~0.011, ~10 nodes per
  wavelength).
- `fem.py`: P1 FEM in torch on the GPU (sparse complex matrix, GMRES with a diagonal preconditioner).

Findings: P1 on the track mesh is 16% (RelL2) from the converged solution; uniform refinements of the same mesh
give 4.3% (38k nodes), 0.93% (151k), vs. 601k nodes (second order, the usual k = 60 pollution error). The
diagonal preconditioner needs 1.2k → 10k GMRES iterations (21 s → 180 s) from 10k to 601k nodes.

Next:
1. Preconditioner: complex-shifted Laplacian `-Δ - (1 + 0.5i)k²` (same boundary conditions), one geometric
   multigrid V-cycle on the nested refinements (linear interpolation, damped Jacobi / Chebyshev smoothing, exact
   LU of the shifted operator on the track mesh), inside right-preconditioned GMRES.
2. Reference: values at the track-mesh nodes from the finest refinements, Richardson-extrapolated (order 2);
   check ~1e-3 or better. Then the baseline (record #0), the figures and the README.
