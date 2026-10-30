"""Record #0 (classical baseline): cell-centered finite volumes on a 3328 x 3328 grid (13 x the output grid,
injection onto the output cell centers), multigrid-preconditioned CG (rtol 1e-9). The inclusion corners limit
FV to order ~4/3, hence the fine grid."""
from speedrun.numerics.fv_diffusion import solve_diffusion

R = 13          # refinement over the 256 x 256 output grid (odd, so output centers are fine-cell centers)


def solve(problem, solver=None):
    n = problem.axes["x"].numel()
    u, _ = solve_diffusion(problem.fns["a"], problem.fns["forcing"], R * n, dtype=problem.dtype,
                           device=problem.device, rtol=1e-9)
    return u[None, R // 2::R, R // 2::R]
