"""Fourier pseudo-spectral + ETDRK4 solvers for periodic problems (Burgers 1D/2D, Kuramoto-Sivashinsky)."""
from __future__ import annotations

import torch

from .fourier import dealias_mask, etdrk4_coefficients, etdrk4_step, wavenumbers


def scalar_conservation_1d(u0, length, N, linear, t_out, substeps, dtype=torch.float64, device="cpu"):
    """u_t = L u - (1/2) (u^2)_x on a periodic interval of given length, N grid points.

    u0: callable on the grid x_j = x0 + length * j / N  (pass values via closure; we call u0(j) with the
        offset-free grid s_j = length * j / N, the caller shifts as needed)
    linear: callable k -> L(k) (diagonal linear operator symbol)
    t_out: 1D tensor of output times, uniformly spaced, starting at 0
    substeps: number of ETDRK4 steps per output interval
    Returns U (len(t_out), N).
    """
    k, m = wavenumbers(N, length, real=True, dtype=dtype, device=device)
    mask = dealias_mask(m, N).to(dtype)
    L = linear(k)
    nt = t_out.numel()
    h = float(t_out[1] - t_out[0]) / substeps
    coefs = etdrk4_coefficients(L, h)
    ik = 1j * k * mask

    def Nfun(v):
        u = torch.fft.irfft(v * mask, n=N)
        return -0.5 * ik * torch.fft.rfft(u * u)

    s = length * torch.arange(N, dtype=dtype, device=device) / N
    v = torch.fft.rfft(u0(s))
    U = torch.empty(nt, N, dtype=dtype, device=device)
    U[0] = torch.fft.irfft(v, n=N)
    for i in range(1, nt):
        for _ in range(substeps):
            v = etdrk4_step(v, Nfun, coefs)
        U[i] = torch.fft.irfft(v, n=N)
    return U


def burgers_2d(u0, v0, length, N, nu, t_out, substeps, dtype=torch.float64, device="cpu", keep_every=1,
               callback=None, stride=1):
    """Vector viscous Burgers  u_t + (u . grad) u = nu Lap u  on a periodic square of side `length`, N x N grid.

    u0, v0: initial fields sampled on the grid (N, N), index [i, j] <-> (x_i, y_j)
    Returns (len(t_out[::keep_every]), 2, N // stride, N // stride) snapshots at t_out[::keep_every],
    restricted to every `stride`-th grid point (injection onto a nested coarser grid).
    """
    kx, mx = wavenumbers(N, length, real=False, dtype=dtype, device=device)
    ky, my = wavenumbers(N, length, real=True, dtype=dtype, device=device)
    KX, KY = kx[:, None], ky[None, :]
    mask = (dealias_mask(mx, N)[:, None] & dealias_mask(my, N)[None, :]).to(dtype)
    L = -nu * (KX**2 + KY**2)
    h = float(t_out[1] - t_out[0]) / substeps
    coefs = etdrk4_coefficients(L, h)
    iKX, iKY = 1j * KX * mask, 1j * KY * mask

    def Nfun(w):
        uh, vh = w[0], w[1]
        irf = lambda a: torch.fft.irfft2(a, s=(N, N))  # noqa: E731
        u, v = irf(uh * mask), irf(vh * mask)
        ux, uy, vx, vy = irf(iKX * uh), irf(iKY * uh), irf(iKX * vh), irf(iKY * vh)
        nu_ = torch.fft.rfft2(-(u * ux + v * uy)) * mask
        nv_ = torch.fft.rfft2(-(u * vx + v * vy)) * mask
        return torch.stack([nu_, nv_])

    w = torch.stack([torch.fft.rfft2(u0), torch.fft.rfft2(v0)])
    nt = t_out.numel()
    out_idx = list(range(0, nt, keep_every))
    out = torch.empty(len(out_idx), 2, N // stride, N // stride, dtype=dtype, device=device)
    o = 0
    for i in range(nt):
        if i > 0:
            for _ in range(substeps):
                w = etdrk4_step(w, Nfun, coefs)
        if i % keep_every == 0:
            snap = torch.stack([torch.fft.irfft2(w[0], s=(N, N)), torch.fft.irfft2(w[1], s=(N, N))])
            out[o] = snap[:, ::stride, ::stride]
            o += 1
            if callback is not None:
                callback(i, float(t_out[i]))
    return out


def navier_stokes_2d(u0, v0, length, N, nu, t_out, substeps, dtype=torch.float64, device="cpu"):
    """Incompressible Navier-Stokes on a periodic square of side `length`, vorticity-streamfunction form
    w_t + u . grad w = nu Lap w  (w = v_x - u_y), N x N grid, 2/3 dealiasing, ETDRK4.

    u0, v0: initial (divergence-free) velocity on the grid (N, N), index [i, j] <-> (x_i, y_j).
    Returns velocity snapshots (len(t_out), 2, N, N) at t_out (uniformly spaced, t_out[0] = initial time).
    """
    kx, mx = wavenumbers(N, length, real=False, dtype=dtype, device=device)
    ky, my = wavenumbers(N, length, real=True, dtype=dtype, device=device)
    KX, KY = kx[:, None], ky[None, :]
    K2 = KX**2 + KY**2
    K2i = torch.where(K2 > 0, 1.0 / torch.where(K2 > 0, K2, torch.ones_like(K2)), torch.zeros_like(K2))
    mask = (dealias_mask(mx, N)[:, None] & dealias_mask(my, N)[None, :]).to(dtype)
    h = float(t_out[1] - t_out[0]) / substeps
    coefs = etdrk4_coefficients(-nu * K2, h)
    irf = lambda a: torch.fft.irfft2(a, s=(N, N))  # noqa: E731
    uh0, vh0 = torch.fft.rfft2(u0), torch.fft.rfft2(v0)
    mean_u, mean_v = uh0[0, 0], vh0[0, 0]                      # mean flow is conserved

    def velocity(wh):
        ph = wh * K2i                                          # Lap psi = -w,  u = psi_y,  v = -psi_x
        uh, vh = 1j * KY * ph, -1j * KX * ph
        uh[0, 0], vh[0, 0] = mean_u, mean_v
        return uh, vh

    def Nfun(wh):
        uh, vh = velocity(wh * mask)
        u, v = irf(uh), irf(vh)
        wx, wy = irf(1j * KX * mask * wh), irf(1j * KY * mask * wh)
        return -torch.fft.rfft2(u * wx + v * wy) * mask

    wh = 1j * KX * vh0 - 1j * KY * uh0
    out = torch.empty(t_out.numel(), 2, N, N, dtype=dtype, device=device)
    for i in range(t_out.numel()):
        if i > 0:
            for _ in range(substeps):
                wh = etdrk4_step(wh, Nfun, coefs)
        uh, vh = velocity(wh)
        out[i, 0], out[i, 1] = irf(uh), irf(vh)
    return out
