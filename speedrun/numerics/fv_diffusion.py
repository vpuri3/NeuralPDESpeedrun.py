"""Cell-centered finite-volume solver for -div(a grad u) = f on the unit square, u = 0 on the boundary.

Discretization: N x N cells, 5-point flux stencil with harmonic averaging of the (cell-wise constant)
coefficient across faces; Dirichlet walls at half-cell distance. Second order for coefficient jumps
aligned with cell faces.

Solver: preconditioned CG with an aggregation multigrid V-cycle (2 x 2 piecewise-constant
aggregation; the Galerkin coarse operator is again a 5-point FV operator whose face
transmissibilities are sums of fine ones), damped-Jacobi smoothing, dense solve on the coarsest grid.
"""
from __future__ import annotations

import torch

from .krylov import pcg


class FVLevel:
    def __init__(self, Tx, Ty):
        self.Tx, self.Ty = Tx, Ty                    # (N+1, N), (N, N+1) face transmissibilities
        self.N = Ty.shape[0]
        self.diag = Tx[:-1] + Tx[1:] + Ty[:, :-1] + Ty[:, 1:]

    def apply(self, u):
        Au = self.diag * u
        Au[1:] -= self.Tx[1:-1] * u[:-1]
        Au[:-1] -= self.Tx[1:-1] * u[1:]
        Au[:, 1:] -= self.Ty[:, 1:-1] * u[:, :-1]
        Au[:, :-1] -= self.Ty[:, 1:-1] * u[:, 1:]
        return Au

    def coarsen(self):
        Tx, Ty = self.Tx, self.Ty
        Txc = Tx[0::2, 0::2] + Tx[0::2, 1::2]       # coarse x-faces at fine x-faces 0, 2, 4, ...
        Tyc = Ty[0::2, 0::2] + Ty[1::2, 0::2]
        return FVLevel(Txc, Tyc)

    def dense(self):
        n = self.N * self.N
        I = torch.eye(n, dtype=self.diag.dtype, device=self.diag.device).reshape(n, self.N, self.N)
        return torch.stack([self.apply(I[k]).reshape(-1) for k in range(n)], dim=1)


class AggregationMG:
    def __init__(self, fine: FVLevel, nu1=2, nu2=2, omega=0.8, coarsest=8):
        self.levels = [fine]
        while self.levels[-1].N % 2 == 0 and self.levels[-1].N > coarsest:
            self.levels.append(self.levels[-1].coarsen())
        self.Ac = self.levels[-1].dense()
        self.nu1, self.nu2, self.omega = nu1, nu2, omega

    def _vcycle(self, l, b):
        L = self.levels[l]
        if l == len(self.levels) - 1:
            return torch.linalg.solve(self.Ac, b.reshape(-1)).reshape(b.shape)
        x = self.omega * b / L.diag
        for _ in range(self.nu1 - 1):
            x = x + self.omega * (b - L.apply(x)) / L.diag
        r = b - L.apply(x)
        rc = r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2] + r[1::2, 1::2]   # R = P^T
        ec = self._vcycle(l + 1, rc)
        e = ec.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)          # P: piecewise constant
        x = x + e
        for _ in range(self.nu2):
            x = x + self.omega * (b - L.apply(x)) / L.diag
        return x

    def __call__(self, b):
        return self._vcycle(0, b)


def transmissibilities(a_cells: torch.Tensor, h: float):
    """Face transmissibilities (already divided by h^2) for cell-wise constant coefficients a (N, N)."""
    N = a_cells.shape[0]
    hm_x = 2.0 * a_cells[:-1] * a_cells[1:] / (a_cells[:-1] + a_cells[1:])
    hm_y = 2.0 * a_cells[:, :-1] * a_cells[:, 1:] / (a_cells[:, :-1] + a_cells[:, 1:])
    Tx = torch.zeros(N + 1, N, dtype=a_cells.dtype, device=a_cells.device)
    Ty = torch.zeros(N, N + 1, dtype=a_cells.dtype, device=a_cells.device)
    Tx[1:-1] = hm_x
    Tx[0], Tx[-1] = 2.0 * a_cells[0], 2.0 * a_cells[-1]
    Ty[:, 1:-1] = hm_y
    Ty[:, 0], Ty[:, -1] = 2.0 * a_cells[:, 0], 2.0 * a_cells[:, -1]
    return Tx / h**2, Ty / h**2


def solve_diffusion(a_fn, f_fn, N, dtype=torch.float64, device="cpu", rtol=1e-10, maxiter=500):
    """Returns cell-center values u (N, N) at x_i = (i + 1/2)/N, and the PCG residual history."""
    h = 1.0 / N
    xc = (torch.arange(N, dtype=dtype, device=device) + 0.5) * h
    X, Y = torch.meshgrid(xc, xc, indexing="ij")
    Tx, Ty = transmissibilities(a_fn(X, Y), h)
    A = FVLevel(Tx, Ty)
    M = AggregationMG(A)
    u, hist = pcg(A.apply, f_fn(X, Y), M, rtol=rtol, maxiter=maxiter)
    return u, hist
