"""Record #0 (classical baseline): Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 directly on the
prescribed N = 256 grid, one ETDRK4 step per snapshot interval."""
import math

from speedrun.numerics.periodic import scalar_conservation_1d

SUBSTEPS = 1


def solve(problem, solver=None):
    nu = problem.params["nu"]
    N = problem.axes["x"].numel()
    u0 = lambda s: problem.fns["u0"](s - math.pi)  # noqa: E731
    U = scalar_conservation_1d(u0, 2 * math.pi, N, lambda k: k**2 - nu * k**4, problem.axes["t"], SUBSTEPS,
                               dtype=problem.dtype, device=problem.device)
    return U[None]
