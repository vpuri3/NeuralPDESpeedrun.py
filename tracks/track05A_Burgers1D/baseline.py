"""Record #0 (classical baseline): Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 directly on the
prescribed N = 1024 grid, SUBSTEPS ETDRK4 steps per snapshot interval."""
from speedrun.numerics.periodic import scalar_conservation_1d

SUBSTEPS = 1


def solve(problem, solver=None):
    nu = problem.params["nu"]
    N = problem.axes["x"].numel()
    U = scalar_conservation_1d(problem.fns["u0"], 2.0, N, lambda k: -nu * k**2, problem.axes["t"], SUBSTEPS,
                               dtype=problem.dtype, device=problem.device)
    return U[None]
