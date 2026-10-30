"""Record #0 (classical baseline): cell-centered finite volumes on the 256 x 256 output grid,
multigrid-preconditioned CG (rtol 1e-8)."""
from speedrun.numerics.fv_diffusion import solve_diffusion

N = 256


def solve(problem):
    u, _ = solve_diffusion(problem.fns["a"], problem.fns["forcing"], N, dtype=problem.dtype,
                           device=problem.device, rtol=1e-8)
    return u[None]
