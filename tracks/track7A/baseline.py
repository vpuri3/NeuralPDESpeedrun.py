"""Record #0 (classical baseline): Chebyshev streamfunction collocation, N x N = 24 x 24,
Newton (with viscosity continuation where needed), dense LU."""
from tracks.cavity_common import chebyshev_solve

N = 24


def solve(problem):
    return chebyshev_solve(problem, N)
