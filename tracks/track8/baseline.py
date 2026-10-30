"""Record #0 (classical baseline): Chebyshev streamfunction collocation, N x N = 16 x 16,
single dense linear solve."""
from tracks.cavity_common import chebyshev_solve

N = 16


def solve(problem):
    return chebyshev_solve(problem, N)
