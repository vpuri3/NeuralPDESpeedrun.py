"""Record #0 (classical baseline): Chebyshev collocation, N + 1 Gauss-Lobatto points, dense solve, barycentric
interpolation to the output grid."""
import torch

from speedrun.numerics import bary_matrix, cheb

N = 32


def chebyshev_collocation(problem, N):
    eps, b = problem.params["eps"], problem.params["b"]
    x, D = cheb(N, 0.0, 1.0, dtype=problem.dtype, device=problem.device)
    A = (-eps * D @ D + b * D)[1:-1, 1:-1]               # homogeneous Dirichlet: drop the boundary unknowns
    u = torch.zeros(N + 1, dtype=problem.dtype, device=problem.device)
    u[1:-1] = torch.linalg.solve(A, problem.fns["forcing"](x[1:-1]))
    return (bary_matrix(x, problem.axes["x"]) @ u)[None]


def solve(problem, state=None):
    return chebyshev_collocation(problem, N)
