"""Record #0 (classical baseline): Chebyshev streamfunction collocation, N x N = 32 x 32,
Newton (with viscosity continuation where needed), dense LU."""
from tracks.cavity_common import chebyshev_solve

N = 32


def solve(problem):
    return chebyshev_solve(problem, N)
