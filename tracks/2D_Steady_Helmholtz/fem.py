"""Linear (P1) finite elements for 2D_Steady_Helmholtz, in torch on the GPU (reference and baseline).

Weak form: int grad r . grad v - k^2 int r v - i k int_circle r v = 0 for all v vanishing on the square, with
r = -exp(-i k x) at the square's nodes. Consistent mass matrices; sparse complex matrix on the device; the system is
solved by restarted GMRES (speedrun.numerics.krylov), preconditioned with the matrix diagonal.
"""
import torch

from speedrun.numerics.krylov import gmres

C128 = torch.complex128


def assemble(points, triangles, outer_edges, k):
    """Sparse (CSR) matrix K - k^2 M - i k B of the mesh, on the points' device."""
    p, t, dev = points.to(torch.float64), triangles.long(), points.device
    x, y = p[t, 0], p[t, 1]                                                     # (m, 3)
    b = torch.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = torch.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area[:, None, None])
    Me = area[:, None, None] / 12 * (torch.ones(3, 3, dtype=torch.float64, device=dev)
                                     + torch.eye(3, dtype=torch.float64, device=dev))
    e = outer_edges.long()
    L = torch.linalg.norm(p[e[:, 1]] - p[e[:, 0]], dim=1)
    Be = L[:, None, None] / 6 * torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64, device=dev)
    rows = torch.cat([t[:, :, None].expand(-1, 3, 3).reshape(-1), e[:, :, None].expand(-1, 2, 2).reshape(-1)])
    cols = torch.cat([t[:, None, :].expand(-1, 3, 3).reshape(-1), e[:, None, :].expand(-1, 2, 2).reshape(-1)])
    vals = torch.cat([(Ke - k * k * Me).reshape(-1).to(C128), (-1j * k * Be).reshape(-1)])
    n = p.shape[0]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce().to_sparse_csr()


def solve(points, triangles, obstacle, outer_edges, k, dirichlet, rtol=1e-10, restart=300, maxiter=20000):
    """Nodal values of r (complex). dirichlet(x, y) -> complex values at the square's nodes."""
    dev = points.device
    Am = assemble(points, triangles, outer_edges, k)
    n = points.shape[0]
    free = torch.ones(n, dtype=torch.bool, device=dev)
    free[obstacle.long()] = False
    ud = torch.zeros(n, dtype=C128, device=dev)
    ud[obstacle.long()] = dirichlet(points[obstacle.long(), 0], points[obstacle.long(), 1]).to(C128)
    rhs = torch.where(free, -(Am @ ud[:, None])[:, 0], 0)
    diag = torch.ones(n, dtype=C128, device=dev)
    crow, col, val = Am.crow_indices(), Am.col_indices(), Am.values()
    rid = torch.repeat_interleave(torch.arange(n, device=dev), crow[1:] - crow[:-1])
    on = rid == col
    diag[rid[on]] = val[on]
    dinv = torch.where(free, 1.0 / diag, 1.0)

    def A_op(v):
        return torch.where(free, (Am @ torch.where(free, v, 0)[:, None])[:, 0], v)

    u, hist = gmres(A_op, rhs, M=lambda v: dinv * v, rtol=rtol, restart=restart, maxiter=maxiter)
    r = torch.where(free, u, 0) + ud
    return r, hist
