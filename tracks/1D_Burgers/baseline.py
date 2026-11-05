"""Record #0 (classical baseline): u(0, x) = -sin(pi x) is odd and 2-periodic, and so is the solution, so
u = 0 on the walls holds automatically on the periodic interval [-1, 1). Fourier pseudo-spectral (2/3
dealiasing) + ETDRK4 on N points, SUBSTEPS steps per output interval; the output grid (dx = 2 / 1000) is every
(N / 1000)-th grid point, plus the periodic image at x = 1."""
import torch

from speedrun.numerics.periodic import scalar_conservation_1d

N = 4000
SUBSTEPS = 1


def solve(problem, state=None):
    nu, u0 = problem.params["nu"], problem.fns["u0"]
    U = scalar_conservation_1d(lambda s: u0(s - 1.0), 2.0, N, lambda k: -nu * k**2, problem.axes["t"], SUBSTEPS,
                               dtype=problem.dtype, device=problem.device)
    U = U[:, :: N // (problem.axes["x"].numel() - 1)]
    return torch.cat([U, U[:, :1]], dim=1)[None]
