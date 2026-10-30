"""Single-element Legendre spectral method for incompressible Navier-Stokes (PN-PN-2), ported from
the MATLAB codes in https://github.com/vpuri3/spec (nsSpectral/, navierStokesSEM/, geom/).

  - velocity on the (N+1)^2 Gauss-Lobatto-Legendre (GLL) grid, pressure on the (N-1)^2 GLL grid of
    degree N-2, 3/2-rule dealiasing of the advection term, weak (Galerkin) form with GLL quadrature
  - curved quadrilateral domains through a map (r, s) in [-1, 1]^2 -> (x, y) (Gordon-Hall in the spec)
  - Dirichlet velocity on the whole boundary

Two solvers share the operators:
  `SEMNavierStokes.steady(...)`    Newton on the full discrete steady system (dense; Stokes in one step)
  `SEMNavierStokes.unsteady(...)`  BDF3/EXT3 with incremental pressure projection, exactly as the spec's
                                   driver.m, viscous and pressure solves by fast diagonalization (affine
                                   (rectangular) domains only)

Arrays are (n_r, n_s): first index along r (x), second along s (y), like MATLAB's ndgrid.
"""
from __future__ import annotations

import math
from typing import Callable

import torch


# ----------------------------------------------------------------------------- 1D building blocks
def gll(N: int, dtype=torch.float64, device="cpu"):
    """N+1 Gauss-Lobatto-Legendre nodes (ascending) and weights on [-1, 1] (spec: semhat/zwgll.m)."""
    z = torch.zeros(N + 1, dtype=dtype, device=device)
    z[0], z[-1] = -1.0, 1.0
    if N > 1:
        i = torch.arange(1, N - 1, dtype=dtype, device=device)
        off = 0.5 * torch.sqrt(i * (i + 2) / ((i + 0.5) * (i + 1.5)))
        M = torch.diag(off, 1) + torch.diag(off, -1)
        z[1:-1] = torch.linalg.eigvalsh(M)
    # weights 2 / (N (N+1) P_N(z)^2)
    p0, p1 = torch.ones_like(z), z.clone()
    for j in range(1, N):
        p0, p1 = p1, ((2 * j + 1) * z * p1 - j * p0) / (j + 1)
    PN = p1 if N > 0 else p0
    return z, 2.0 / (N * (N + 1) * PN**2)


def interp_mat(xo: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Lagrange interpolation matrix from nodes xi to points xo (spec: semhat/interp_mat.m)."""
    d = xi[:, None] - xi[None, :]
    a = 1.0 / torch.where(torch.eye(len(xi), dtype=torch.bool, device=xi.device), torch.ones_like(d), d).prod(1)
    X = xo[:, None] - xi[None, :]
    hit = X.abs() < 1e-14
    Xs = torch.where(hit, torch.ones_like(X), X)
    J = a[None, :] / Xs
    J = J / J.sum(1, keepdim=True)
    return torch.where(hit.any(1, keepdim=True), hit.to(J.dtype), J)


def deriv_mat(z: torch.Tensor) -> torch.Tensor:
    """Collocation derivative matrix on nodes z (spec: semhat/dhat.m)."""
    n = len(z)
    d = z[:, None] - z[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    a = 1.0 / torch.where(eye, torch.ones_like(d), d).prod(1)        # barycentric weights
    D = (a[None, :] / a[:, None]) / torch.where(eye, torch.ones_like(d), d)
    D = torch.where(eye, torch.zeros_like(D), D)
    return D - torch.diag(D.sum(1))


def bdfext_coeffs(order: int, dt: float):
    """Constant-step BDFk/EXTk: du/dt ~ sum_i b[i] u^{n+1-i},  explicit terms ~ sum_i a[i] g^{n-i}."""
    a = {1: [1.0], 2: [2.0, -1.0], 3: [3.0, -3.0, 1.0]}[order]
    b = {1: [1.0, -1.0], 2: [1.5, -2.0, 0.5], 3: [11 / 6, -3.0, 1.5, -1 / 3]}[order]
    a = a + [0.0] * (3 - len(a))
    b = [c / dt for c in b] + [0.0] * (4 - len(b))
    return a, b


def gordon_hall(bottom: Callable, top: Callable, left: Callable, right: Callable):
    """Transfinite (Gordon-Hall) map from [-1, 1]^2 given the four edges as callables t -> (x, y):
    bottom(r) = edge s=-1, top(r) = s=+1, left(s) = r=-1, right(s) = r=+1 (spec: geom/gordonhall2d.m)."""
    def m(r, s):
        one = torch.ones_like(r)
        out = []
        for k in range(2):
            e = (0.5 * (1 - s) * bottom(r)[k] + 0.5 * (1 + s) * top(r)[k]
                 + 0.5 * (1 - r) * left(s)[k] + 0.5 * (1 + r) * right(s)[k])
            v = (0.25 * (1 - r) * (1 - s) * bottom(-one)[k] + 0.25 * (1 + r) * (1 - s) * bottom(one)[k]
                 + 0.25 * (1 - r) * (1 + s) * top(-one)[k] + 0.25 * (1 + r) * (1 + s) * top(one)[k])
            out.append(e - v)
        return out[0], out[1]
    return m


def rectangle_map(x0, x1, y0, y1):
    return lambda r, s: (x0 + 0.5 * (x1 - x0) * (r + 1), y0 + 0.5 * (y1 - y0) * (s + 1))


# ----------------------------------------------------------------------------- the solver
class _Grid:
    """Nodes, derivative matrix, geometric factors and mass on one tensor GLL grid."""

    def __init__(self, n, xmap, dtype, device):
        self.z, self.w = gll(n - 1, dtype, device)
        self.D = deriv_mat(self.z)
        R, S = torch.meshgrid(self.z, self.z, indexing="ij")
        self.x, self.y = xmap(R, S)
        xr, xs = self.D @ self.x, self.x @ self.D.T
        yr, ys = self.D @ self.y, self.y @ self.D.T
        self.J = xr * ys - xs * yr
        self.rx, self.ry, self.sx, self.sy = ys / self.J, -xs / self.J, -yr / self.J, xr / self.J
        self.B = self.J * (self.w[:, None] * self.w[None, :])


class SEMNavierStokes:
    """PN-PN-2 Legendre spectral discretization of  u_t + u.grad u + grad p = nu Lap u,  div u = 0
    on the image of [-1, 1]^2 under `xmap`, Dirichlet velocity on the whole boundary."""

    def __init__(self, N: int, xmap: Callable, dtype=torch.float64, device="cpu"):
        self.N, self.xmap, self.dtype, self.device = N, xmap, dtype, device
        n1, n2 = N + 1, N - 1
        nd = math.ceil(1.5 * n1)
        self.g1 = _Grid(n1, xmap, dtype, device)       # velocity
        self.g2 = _Grid(n2, xmap, dtype, device)       # pressure
        self.gd = _Grid(nd, xmap, dtype, device)       # dealiasing (quadrature for advection)
        self.J1d = interp_mat(self.gd.z, self.g1.z)
        self.J21 = interp_mat(self.g1.z, self.g2.z)
        g = self.g1
        self.g11 = g.B * (g.rx * g.rx + g.ry * g.ry)
        self.g12 = g.B * (g.rx * g.sx + g.ry * g.sy)
        self.g22 = g.B * (g.sx * g.sx + g.sy * g.sy)
        self.mask = torch.zeros(n1, n1, dtype=dtype, device=device)
        self.mask[1:-1, 1:-1] = 1.0

    # ---- operators (weak form; spec nsSpectral/*.m)
    def grad(self, u):
        g = self.g1
        ur, us = g.D @ u, u @ g.D.T
        return g.rx * ur + g.sx * us, g.ry * ur + g.sy * us

    def lapl(self, u):                                   # (grad v, grad u)
        D = self.g1.D
        ur, us = D @ u, u @ D.T
        return D.T @ (self.g11 * ur + self.g12 * us) + (self.g12 * ur + self.g22 * us) @ D

    def advect(self, u, cx, cy):                          # (v, c . grad u), dealiased
        J = self.J1d
        ux, uy = self.grad(u)
        d = lambda f: J @ f @ J.T  # noqa: E731
        return J.T @ (self.gd.B * (d(cx) * d(ux) + d(cy) * d(uy))) @ J

    def vgradp(self, p):                                  # (div v, p)
        g, J = self.g1, self.J21
        BJp = g.B * (J @ p @ J.T)
        return (g.D.T @ (g.rx * BJp) + (g.sx * BJp) @ g.D,
                g.D.T @ (g.ry * BJp) + (g.sy * BJp) @ g.D)

    def diver(self, ux, uy):                              # (q, div u)
        J = self.J21
        uxx, _ = self.grad(ux)
        _, uyy = self.grad(uy)
        return J.T @ (self.g1.B * (uxx + uyy)) @ J

    def boundary_lift(self, u_bc, v_bc, t=None):
        g = self.g1
        args = (g.x, g.y) if t is None else (g.x, g.y, t)
        return u_bc(*args) * (1 - self.mask), v_bc(*args) * (1 - self.mask)

    # ---- steady: Newton on the full discrete system
    def steady(self, nu: float, u_bc, v_bc, stokes=False, tol=1e-11, maxit=30, chunk=None, verbose=False):
        """Solve nu A u + C(u) u - D^T p = 0, D u = 0 (C = 0 for Stokes) for interior velocity + pressure.
        The pressure constant is fixed by a bordered row int p = 0."""
        n1, n2 = self.N + 1, self.N - 1
        ni = (n1 - 2) ** 2
        ub, vb = self.boundary_lift(u_bc, v_bc)
        pad = lambda h: torch.nn.functional.pad(h.reshape(n1 - 2, n1 - 2), (1, 1, 1, 1))  # noqa: E731

        def residual(z):
            u, v = pad(z[:ni]) + ub, pad(z[ni:2 * ni]) + vb
            p = z[2 * ni:].reshape(n2, n2)
            px, py = self.vgradp(p)
            ru, rv = nu * self.lapl(u) - px, nu * self.lapl(v) - py
            if not stokes:
                ru, rv = ru + self.advect(u, u, v), rv + self.advect(v, u, v)
            return torch.cat([ru[1:-1, 1:-1].reshape(-1), rv[1:-1, 1:-1].reshape(-1),
                              self.diver(u, v).reshape(-1)])

        nz = 2 * ni + n2 * n2
        c = torch.zeros(nz + 1, dtype=self.dtype, device=self.device)
        c[2 * ni:nz] = self.g2.B.reshape(-1)
        z = torch.zeros(nz, dtype=self.dtype, device=self.device)
        for it in range(1 if stokes else maxit):
            r = residual(z)
            Jm = (torch.func.jacfwd(residual)(z) if chunk is None    # forward mode: no autograd thread
                  else torch.func.jacrev(residual, chunk_size=chunk)(z))
            K = torch.zeros(nz + 1, nz + 1, dtype=self.dtype, device=self.device)
            K[:nz, :nz] = Jm
            K[nz], K[:, nz] = c, c
            dz = torch.linalg.solve(K, -torch.cat([r, r.new_zeros(1)]))[:nz]
            z = z + dz
            step = float(dz.abs().max())
            if verbose:
                print(f"    newton {it}: |r|={float(r.abs().max()):.3e} |dz|={step:.3e}")
            if step < tol:
                break
        self.u, self.v = pad(z[:ni]) + ub, pad(z[ni:2 * ni]) + vb
        self.p = z[2 * ni:].reshape(n2, n2)
        return self.u, self.v

    # ---- steady Stokes, matrix-free: Uzawa (pressure Schur complement) CG, affine domains
    def stokes_uzawa(self, nu: float, u_bc, v_bc, rtol=1e-12, maxiter=2000):
        """nu A u - D^T p = 0, D u = 0: CG on S p = D H^-1 D^T p (H = nu A, inverted by fast diagonalization),
        preconditioned by the inverse pressure mass matrix (S is spectrally equivalent to it / nu)."""
        from .krylov import pcg
        self._fdm_setup()
        ub, vb = self.boundary_lift(u_bc, v_bc)
        Hi = lambda r: self._visc_solve(r, 0.0, nu)  # noqa: E731
        fu, fv = -nu * self.lapl(ub), -nu * self.lapl(vb)

        def vel(p, with_f=True):
            px, py = self.vgradp(p)
            if with_f:
                px, py = px + fu, py + fv
            return Hi(px), Hi(py)

        B2 = self.g2.B
        proj = lambda p: p - (B2 * p).sum() / B2.sum()  # noqa: E731
        S = lambda p: proj(self.diver(*vel(p, False)))  # noqa: E731  (SPD on mean-free pressures)
        uh0, vh0 = vel(torch.zeros_like(B2))
        rhs = proj(-self.diver(uh0 + ub, vh0 + vb))
        p, hist = pcg(S, rhs, M=lambda r: proj(nu * r / B2), rtol=rtol, maxiter=maxiter)
        uh, vh = vel(p)
        self.u, self.v, self.p = uh + ub, vh + vb, p
        self.cg_iters = len(hist)
        return self.u, self.v

    # ---- unsteady: BDF3/EXT3 + incremental pressure projection (spec driver.m), affine domains
    def _fdm_setup(self):
        g = self.g1
        hx = float(g.x[-1, 0] - g.x[0, 0]) / 2
        hy = float(g.y[0, -1] - g.y[0, 0]) / 2
        assert torch.allclose(g.J, torch.full_like(g.J, hx * hy)), "fast diagonalization needs a rectangle"
        D, w = g.D, g.w

        def gen_eig(A, B):                              # A S = B S L,  S^T B S = I
            Bh = torch.diag(B.diagonal().rsqrt()) if torch.equal(B, torch.diag(B.diagonal())) else None
            if Bh is not None:
                L, Q = torch.linalg.eigh(Bh @ A @ Bh)
                return Bh @ Q, L
            Lc = torch.linalg.cholesky(B)
            Li = torch.linalg.inv(Lc)
            L, Q = torch.linalg.eigh(Li @ A @ Li.T)
            return Li.T @ Q, L

        fd = {}
        for k, h in (("x", hx), ("y", hy)):
            Bv = h * torch.diag(w)
            Dv = D / h
            Av = Dv.T @ Bv @ Dv
            S, L = gen_eig(Av[1:-1, 1:-1], Bv[1:-1, 1:-1])
            fd[k + "v"] = (S, L)
            # pressure: E = Axp (x) Byp + Bxp (x) Ayp with E = D B^-1 M D^T (spec driver.m)
            M = torch.diag(torch.cat([w.new_zeros(1), torch.ones_like(w[1:-1]), w.new_zeros(1)]))
            Bi = torch.diag(1.0 / (h * w))
            J = self.J21
            Ap = J.T @ Bv @ (Dv @ (M @ Bi @ M) @ Dv.T) @ Bv @ J
            Bp = J.T @ Bv @ (M @ Bi @ M) @ Bv @ J
            fd[k + "p"] = gen_eig(0.5 * (Ap + Ap.T), 0.5 * (Bp + Bp.T))
        Lp = fd["xp"][1][:, None] + fd["yp"][1][None, :]
        # invert all modes except the single null mode (constant pressure). A relative threshold would also
        # drop the lowest physical modes at large N, where max |Lp| ~ N^4.
        null = Lp.abs().argmin()
        Lp = Lp.reshape(-1).clone()
        Lp[null] = 1.0
        self._Lpi = (1.0 / Lp).index_fill(0, null.reshape(1), 0.0).reshape(self.N - 1, self.N - 1)
        self._fd = fd

    def _visc_solve(self, b, b0, nu):
        (Sx, Lx), (Sy, Ly) = self._fd["xv"], self._fd["yv"]
        bi = b[1:-1, 1:-1]
        u = Sx @ ((Sx.T @ bi @ Sy) / (b0 + nu * (Lx[:, None] + Ly[None, :]))) @ Sy.T
        return torch.nn.functional.pad(u, (1, 1, 1, 1))

    def _pres_solve(self, g):
        Sx, Sy = self._fd["xp"][0], self._fd["yp"][0]
        return Sx @ ((Sx.T @ g @ Sy) * self._Lpi) @ Sy.T

    def unsteady(self, nu: float, u0, v0, u_bc, v_bc, t_out, dt: float, callback=None):
        """March from t = 0 (initial velocity u0, v0 callables of (x, y)) and return velocity snapshots at
        the times t_out (multiples of dt), shape (2, len(t_out), N+1, N+1) on the GLL grid."""
        self._fdm_setup()
        g = self.g1
        u, v = u0(g.x, g.y), v0(g.x, g.y)
        n2 = self.N - 1
        p, p1 = torch.zeros(n2, n2, dtype=self.dtype, device=self.device), None
        hist_u, hist_v, hist_gu, hist_gv = [u], [v], [], []
        steps = [int(round(float(t) / dt)) for t in t_out]
        assert all(abs(s * dt - float(t)) < 1e-9 for s, t in zip(steps, t_out)), "t_out must be multiples of dt"
        out = torch.empty(2, len(steps), *u.shape, dtype=self.dtype, device=self.device)
        for k, s in enumerate(steps):
            if s == 0:
                out[0, k], out[1, k] = u, v
        ub, vb = self.boundary_lift(u_bc, v_bc)
        B = g.B
        for n in range(1, max(steps) + 1):
            a, b = bdfext_coeffs(min(n, 3), dt)
            hist_gu.insert(0, -self.advect(u, u, v))
            hist_gv.insert(0, -self.advect(v, u, v))
            del hist_gu[3:], hist_gv[3:]
            pe = 2 * p - p1 if p1 is not None else p
            px, py = self.vgradp(pe)
            rhs = []
            for hist_g, hist, lift, pg in ((hist_gu, hist_u, ub, px), (hist_gv, hist_v, vb, py)):
                r = sum(a[i] * hist_g[i] for i in range(len(hist_g)))
                r = r - B * sum(b[i + 1] * hist[i] for i in range(len(hist)))
                r = r - (nu * self.lapl(lift) + b[0] * B * lift) + pg
                rhs.append(r)
            us = self._visc_solve(rhs[0], b[0], nu) + ub
            vs = self._visc_solve(rhs[1], b[0], nu) + vb
            # pressure projection
            dp = b[0] * self._pres_solve(-self.diver(us, vs))
            gx, gy = self.vgradp(dp)
            u = us + self.mask * gx / (b[0] * B)
            v = vs + self.mask * gy / (b[0] * B)
            p1, p = p, pe + dp
            hist_u.insert(0, u), hist_v.insert(0, v)
            del hist_u[3:], hist_v[3:]
            for k, s in enumerate(steps):
                if s == n:
                    out[0, k], out[1, k] = u, v
            if callback is not None:
                callback(n, n * dt, u, v)
        self.u, self.v, self.p = u, v, p
        return out

    # ---- evaluation
    def interp_rs(self, f, r, s):
        """Interpolate nodal field(s) f (..., N+1, N+1) to the tensor grid of reference points r x s."""
        Jr, Js = interp_mat(r, self.g1.z), interp_mat(s, self.g1.z)
        return Jr @ f @ Js.T

    def interp_rect(self, f, x, y):
        """Interpolate to the tensor grid x x y (rectangular domains only)."""
        g = self.g1
        x0, x1, y0, y1 = g.x[0, 0], g.x[-1, 0], g.y[0, 0], g.y[0, -1]
        return self.interp_rs(f, 2 * (x - x0) / (x1 - x0) - 1, 2 * (y - y0) / (y1 - y0) - 1)
