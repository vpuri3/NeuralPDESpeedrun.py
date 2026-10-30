"""Record #0 (classical baseline): Chebyshev collocation, N = 16, dense solve."""
import torch

from speedrun.numerics import bary_matrix, cheb

N = 16


def solve(problem):
    x, D = cheb(N, 0.0, 1.0, dtype=problem.dtype, device=problem.device)
    A = D @ D @ D
    b = problem.fns["forcing"](x)
    A[0] = 0.0
    A[0, 0] = 1.0
    b[0] = problem.fns["u_left"]
    A[-1] = 0.0
    A[-1, -1] = 1.0
    b[-1] = problem.fns["u_right"]
    A[-2] = D[-1]
    b[-2] = problem.fns["du_right"]
    u = torch.linalg.solve(A, b)
    return (bary_matrix(x, problem.axes["x"]) @ u)[None]
