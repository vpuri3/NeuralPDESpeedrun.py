"""Chebyshev-collocation streamfunction solver for steady 2D incompressible (Navier-)Stokes on a rectangle.

    nu * Lap^2 psi - psi_y * (Lap psi)_x + psi_x * (Lap psi)_y = 0          (Navier-Stokes, vorticity form)
    nu * Lap^2 psi = 0                                                      (Stokes, `stokes=True`)

with u = psi_y, v = -psi_x and Dirichlet velocity data on the boundary, which gives
psi (from the boundary flux integral) and d(psi)/dn on the boundary.

Discretization: tensor Chebyshev-Gauss-Lobatto grid (Nx+1) x (Ny+1), unknowns psi at all nodes.
Rows: psi = g on the 4 boundary sides; the normal-derivative condition on the boundary is imposed
through the rows of the first interior ring ("row replacement"); the PDE at the remaining interior
nodes. Newton's method with a dense LU (problem sizes here are a few thousand unknowns), optional
continuation in the viscosity.
"""
from __future__ import annotations

import torch

from .cheb import bary_matrix, cheb


class StreamfunctionSolver:
    def __init__(self, Nx, Ny, xlim, ylim, u_bc, v_bc, dtype=torch.float64, device="cpu", n_flux=200001):
        self.Nx, self.Ny = Nx, Ny
        self.xlim, self.ylim = xlim, ylim
        self.dtype, self.device = dtype, device
        x, Dx1 = cheb(Nx, *xlim, dtype=dtype, device=device)
        y, Dy1 = cheb(Ny, *ylim, dtype=dtype, device=device)
        self.x, self.y = x, y
        Ix = torch.eye(Nx + 1, dtype=dtype, device=device)
        Iy = torch.eye(Ny + 1, dtype=dtype, device=device)
        self.D1x, self.D1y = Dx1, Dy1
        D2x, D2y = Dx1 @ Dx1, Dy1 @ Dy1
        D3x, D3y = D2x @ Dx1, D2y @ Dy1
        D4x, D4y = D2x @ D2x, D2y @ D2y
        # flattened index (i, j) -> i * (Ny+1) + j  ; x varies along i
        self.Dx = torch.kron(Dx1, Iy)
        self.Dy = torch.kron(Ix, Dy1)
        self.Lap = torch.kron(D2x, Iy) + torch.kron(Ix, D2y)
        self.B = torch.kron(D4x, Iy) + 2.0 * torch.kron(D2x, D2y) + torch.kron(Ix, D4y)
        self.DxL = torch.kron(D3x, Iy) + torch.kron(Dx1, D2y)
        self.DyL = torch.kron(D2x, Dy1) + torch.kron(Ix, D3y)
        self.n = (Nx + 1) * (Ny + 1)
        self._setup_bc(u_bc, v_bc, n_flux)

    # ------------------------------------------------------------------ boundary data
    def _boundary_psi(self, u_bc, v_bc, n_flux):
        """psi along the boundary from the flux integral d(psi) = u dy - v dx (counter-clockwise),
        evaluated at the Chebyshev boundary nodes. psi(x0, y0) = 0."""
        (x0, x1), (y0, y1) = self.xlim, self.ylim
        dt, dev = self.dtype, self.device
        s = torch.linspace(0.0, 1.0, n_flux, dtype=dt, device=dev)

        def cumtrap(f, ds):
            c = torch.zeros_like(f)
            c[1:] = torch.cumsum(0.5 * (f[1:] + f[:-1]) * ds, dim=0)
            return c

        def interp(sq, c):                                  # linear interpolation on the uniform s-grid
            pos = sq * (n_flux - 1)
            i = torch.clamp(torch.floor(pos).to(torch.int64), 0, n_flux - 2)
            w = pos - i.to(dt)
            return c[i] * (1 - w) + c[i + 1] * w

        # bottom: (x0->x1, y0): dpsi = -v dx
        xb = x0 + (x1 - x0) * s
        cb = cumtrap(-v_bc(xb, torch.full_like(xb, y0)), (x1 - x0) / (n_flux - 1))
        # right: (x1, y0->y1): dpsi = u dy
        yr = y0 + (y1 - y0) * s
        cr = cb[-1] + cumtrap(u_bc(torch.full_like(yr, x1), yr), (y1 - y0) / (n_flux - 1))
        # top: (x1->x0, y1): dpsi = -v dx, dx < 0
        xt = x1 - (x1 - x0) * s
        ct = cr[-1] + cumtrap(v_bc(xt, torch.full_like(xt, y1)), (x1 - x0) / (n_flux - 1))
        # left: (x0, y1->y0): dpsi = u dy, dy < 0
        yl = y1 - (y1 - y0) * s
        cl = ct[-1] + cumtrap(-u_bc(torch.full_like(yl, x0), yl), (y1 - y0) / (n_flux - 1))
        self.flux_mismatch = float(cl[-1].abs())

        x, y = self.x, self.y
        psi_bottom = interp((x - x0) / (x1 - x0), cb)
        psi_right = interp((y - y0) / (y1 - y0), cr)
        psi_top = interp((x1 - x) / (x1 - x0), ct)
        psi_left = interp((y1 - y) / (y1 - y0), cl)
        return psi_bottom, psi_right, psi_top, psi_left

    def _setup_bc(self, u_bc, v_bc, n_flux):
        Nx, Ny = self.Nx, self.Ny
        idx = lambda i, j: i * (Ny + 1) + j  # noqa: E731
        x, y = self.x, self.y
        pb, pr, pt, pl = self._boundary_psi(u_bc, v_bc, n_flux)
        rows, vals, kinds = [], [], []          # kinds: 0 = Dirichlet, 1 = d/dx, 2 = d/dy
        target = []                             # row index that this condition replaces
        # Dirichlet on all boundary nodes
        for i in range(Nx + 1):
            for j in (0, Ny):
                target.append(idx(i, j)); rows.append(idx(i, j)); kinds.append(0)
                vals.append(pb[i] if j == 0 else pt[i])
        for j in range(1, Ny):
            for i in (0, Nx):
                target.append(idx(i, j)); rows.append(idx(i, j)); kinds.append(0)
                vals.append(pl[j] if i == 0 else pr[j])
        # Neumann via first interior ring
        # x-sides: ring nodes (1, j), (Nx-1, j), j = 1..Ny-1  ->  psi_x = -v at (0, j), (Nx, j)
        for j in range(1, Ny):
            for (ir, ib) in ((1, 0), (Nx - 1, Nx)):
                target.append(idx(ir, j)); rows.append(idx(ib, j)); kinds.append(1)
                vals.append(-v_bc(x[ib:ib + 1], y[j:j + 1])[0])
        # y-sides: ring nodes (i, 1), (i, Ny-1), i = 2..Nx-2  ->  psi_y = u at (i, 0), (i, Ny)
        for i in range(2, Nx - 1):
            for (jr, jb) in ((1, 0), (Ny - 1, Ny)):
                target.append(idx(i, jr)); rows.append(idx(i, jb)); kinds.append(2)
                vals.append(u_bc(x[i:i + 1], y[jb:jb + 1])[0])
        dev = self.device
        self.bc_target = torch.tensor(target, dtype=torch.int64, device=dev)
        self.bc_row = torch.tensor(rows, dtype=torch.int64, device=dev)
        self.bc_kind = torch.tensor(kinds, dtype=torch.int64, device=dev)
        self.bc_val = torch.stack([torch.as_tensor(v, dtype=self.dtype) for v in vals]).to(dev)
        I = torch.eye(self.n, dtype=self.dtype, device=dev)
        ops = torch.stack([I[self.bc_row], self.Dx[self.bc_row], self.Dy[self.bc_row]])   # (3, nbc, n)
        self.bc_mat = ops[self.bc_kind, torch.arange(len(rows), device=dev)]                # (nbc, n)

    # ------------------------------------------------------------------ Newton
    def residual_and_jacobian(self, psi, nu, stokes):
        F = nu * (self.B @ psi)
        J = nu * self.B.clone()
        if not stokes:
            px, py = self.Dx @ psi, self.Dy @ psi
            Lx, Ly = self.DxL @ psi, self.DyL @ psi
            F = F - py * Lx + px * Ly
            J = J - Lx[:, None] * self.Dy - py[:, None] * self.DxL + Ly[:, None] * self.Dx + px[:, None] * self.DyL
        F[self.bc_target] = self.bc_mat @ psi - self.bc_val
        J[self.bc_target] = self.bc_mat
        return F, J

    def solve(self, nu, stokes=False, psi0=None, nu_path=None, tol=1e-11, maxit=30, verbose=False):
        psi = torch.zeros(self.n, dtype=self.dtype, device=self.device) if psi0 is None else psi0.clone()
        path = list(nu_path or []) + [nu]
        self.newton_hist = []
        for nu_k in path:
            for it in range(maxit):
                F, J = self.residual_and_jacobian(psi, nu_k, stokes)
                s = 1.0 / J.abs().amax(dim=1)                 # row equilibration (PDE rows ~ N^8, BC rows ~ N^2)
                dpsi = torch.linalg.solve(J * s[:, None], F * s)
                psi = psi - dpsi
                step = float(dpsi.abs().max()) / max(float(psi.abs().max()), 1e-30)
                self.newton_hist.append((nu_k, it, step))
                if verbose:
                    print(f"  nu={nu_k:.3e} it={it} |dpsi|/|psi|={step:.3e}")
                if step < tol or stokes:
                    break
        self.psi = psi
        return psi

    # ------------------------------------------------------------------ outputs
    def velocity_on_grid(self, psi=None):
        psi = self.psi if psi is None else psi
        u = (self.Dy @ psi).reshape(self.Nx + 1, self.Ny + 1)
        v = -(self.Dx @ psi).reshape(self.Nx + 1, self.Ny + 1)
        return u, v

    def evaluate(self, xe, ye, psi=None):
        """(u, v) on the tensor grid xe x ye via barycentric interpolation. Returns (2, len(xe), len(ye))."""
        u, v = self.velocity_on_grid(psi)
        Px, Py = bary_matrix(self.x, xe), bary_matrix(self.y, ye)
        return torch.stack([Px @ u @ Py.T, Px @ v @ Py.T])
