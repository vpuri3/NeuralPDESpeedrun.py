"""Record #0 (classical baseline): 2D Chebyshev collocation (tensor Gauss-Lobatto grid, N x N), the
Kronecker-assembled operator solved with a dense LU, barycentric interpolation to the output grid."""
import torch

from speedrun.numerics import bary_matrix, cheb

N = 96


def chebyshev_collocation(problem, N):
    eps = problem.params["eps"]
    bx, by = problem.params["b"]
    dt, dev = problem.dtype, problem.device
    x, D = cheb(N, 0.0, 1.0, dtype=dt, device=dev)
    Di, D2i = D[1:-1, 1:-1], (D @ D)[1:-1, 1:-1]        # homogeneous Dirichlet: drop boundary unknowns
    I = torch.eye(N - 1, dtype=dt, device=dev)
    A = torch.kron(-eps * D2i + bx * Di, I) + torch.kron(I, -eps * D2i + by * Di)
    xi = x[1:-1]
    Xi, Yi = torch.meshgrid(xi, xi, indexing="ij")
    f = problem.fns["forcing"](Xi, Yi).reshape(-1)
    U = torch.zeros(N + 1, N + 1, dtype=dt, device=dev)
    U[1:-1, 1:-1] = torch.linalg.solve(A, f).reshape(N - 1, N - 1)
    Px, Py = bary_matrix(x, problem.axes["x"]), bary_matrix(x, problem.axes["y"])
    return (Px @ U @ Py.T)[None]


def solve(problem, state=None):
    return chebyshev_collocation(problem, N)
