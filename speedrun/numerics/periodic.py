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
