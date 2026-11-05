"""Record #0 (classical baseline): Chebyshev collocation on one cell, [0, 1] x [0, 1] (N x N Gauss-Lobatto grid,
Dirichlet in x, Neumann in y, Kronecker-assembled operator, dense LU). The solution is even about y = 1 (the
problem is invariant under y -> 2 - y), so the cell [1, 2] is its mirror image. Barycentric interpolation to the
output grid."""
import torch

from speedrun.numerics import bary_matrix, cheb

N = 48


def chebyshev_cell(problem, N, M=None):
    """Collocation solution T on the (N+1) x (M+1) Chebyshev grid of the cell [0, 1]^2; returns (x, y, T)."""
    M = M or N
    pe = problem.params["Pe"]
    dt, dev = problem.dtype, problem.device
    x, Dx = cheb(N, 0.0, 1.0, dtype=dt, device=dev)
    y, Dy = cheb(M, 0.0, 1.0, dtype=dt, device=dev)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u, v = problem.fns["velocity"](X, Y)
    Ix, Iy = torch.eye(N + 1, dtype=dt, device=dev), torch.eye(M + 1, dtype=dt, device=dev)
    DX, DY = torch.kron(Dx, Iy), torch.kron(Ix, Dy)
    A = (torch.kron(Dx @ Dx, Iy) + torch.kron(Ix, Dy @ Dy)) / pe - u.reshape(-1, 1) * DX - v.reshape(-1, 1) * DY
    rhs = torch.zeros((N + 1) * (M + 1), dtype=dt, device=dev)
    idx = torch.arange((N + 1) * (M + 1), device=dev).reshape(N + 1, M + 1)
    for j in (0, M):                                   # insulated walls y = 0, 1: T_y = 0
        A[idx[:, j]] = DY[idx[:, j]]
    for i in (0, N):                                   # T(0, y) = 0, T(1, y) = 1 (corners included)
        A[idx[i]] = 0.0
        A[idx[i], idx[i]] = 1.0
        rhs[idx[i]] = problem.fns["bc"](X[i], Y[i])
    T = torch.linalg.solve(A, rhs).reshape(N + 1, M + 1)
    return x, y, T


def to_output(problem, x, y, T):
    xe, ye = problem.axes["x"], problem.axes["y"]
    ym = torch.where(ye > 1.0, 2.0 - ye, ye)            # mirror the upper cell onto [0, 1]
    return (bary_matrix(x, xe) @ T @ bary_matrix(y, ym).T)[None]


def solve(problem, state=None):
    return to_output(problem, *chebyshev_cell(problem, N))
