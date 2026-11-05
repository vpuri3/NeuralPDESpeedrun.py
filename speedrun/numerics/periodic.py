"""Fourier pseudo-spectral + ETDRK4 solvers for periodic problems (Burgers 1D, Kuramoto-Sivashinsky)."""
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

