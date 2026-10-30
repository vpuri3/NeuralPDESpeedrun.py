"""Fourier / sine pseudo-spectral utilities and the ETDRK4 integrator."""
from __future__ import annotations

import math

import torch


def wavenumbers(N: int, L: float, real: bool = False, dtype=torch.float64, device="cpu"):
    """Angular wavenumbers 2*pi*m/L for an N-point periodic grid of length L (fft or rfft layout)."""
    m = (torch.fft.rfftfreq(N, d=1.0 / N, dtype=dtype, device=device) if real
         else torch.fft.fftfreq(N, d=1.0 / N, dtype=dtype, device=device))
    return 2.0 * math.pi * m / L, m


def dealias_mask(m: torch.Tensor, N: int) -> torch.Tensor:
    """2/3-rule mask on integer mode numbers m: keep |m| < N/3."""
    return (m.abs() < N / 3.0)


def etdrk4_coefficients(L: torch.Tensor, h: float, M: int = 32):
    """Kassam & Trefethen (2005) ETDRK4 coefficients for u_t = L u + N(u) with diagonal L (real or complex).

    Returns (E, E2, Q, f1, f2, f3) with the same shape as L.
    """
    cdt = torch.complex128
    Lc = L.to(cdt)
    r = torch.exp(1j * math.pi * (torch.arange(1, M + 1, dtype=torch.float64, device=L.device) - 0.5) / M).to(cdt)
    LR = h * Lc.unsqueeze(-1) + r
    E = torch.exp(h * Lc)
    E2 = torch.exp(h * Lc / 2)
    eLR = torch.exp(LR)
    Q = h * ((torch.exp(LR / 2) - 1.0) / LR).mean(dim=-1)
    f1 = h * ((-4.0 - LR + eLR * (4.0 - 3.0 * LR + LR**2)) / LR**3).mean(dim=-1)
    f2 = h * ((2.0 + LR + eLR * (-2.0 + LR)) / LR**3).mean(dim=-1)
    f3 = h * ((-4.0 - 3.0 * LR - LR**2 + eLR * (4.0 - LR)) / LR**3).mean(dim=-1)
    if not L.is_complex():
        E, E2, Q, f1, f2, f3 = (c.real for c in (E, E2, Q, f1, f2, f3))
    return E, E2, Q, f1, f2, f3


def etdrk4_step(v, Nfun, coefs):
    """One ETDRK4 step for spectral state v (tensor or tuple of tensors sharing the same L)."""
    E, E2, Q, f1, f2, f3 = coefs
    Nv = Nfun(v)
    a = E2 * v + Q * Nv
    Na = Nfun(a)
    b = E2 * v + Q * Na
    Nb = Nfun(b)
    c = E2 * a + Q * (2.0 * Nb - Nv)
    Nc = Nfun(c)
    return E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3


# ----------------------------------------------------------------------------- sine transforms (DST-I)
def _dst_raw(f: torch.Tensor, dim: int) -> torch.Tensor:
    """S_k = sum_{j=1}^{N-1} f_j sin(pi j k / N), k = 1..N-1, along `dim` (length N-1). Works for complex f."""
    f = f.movedim(dim, -1) if hasattr(f, "movedim") else f
    n1 = f.shape[-1]
    N = n1 + 1
    z = torch.zeros(*f.shape[:-1], 1, dtype=f.dtype, device=f.device)
    g = torch.cat([z, f, z, -torch.flip(f, dims=[-1])], dim=-1)          # odd extension, length 2N
    F = torch.fft.fft(g, dim=-1)[..., 1:N]
    S = F * (0.5j)                                                         # F_k = -2i S_k
    if not f.is_complex():
        S = S.real
    return S.movedim(-1, dim) if hasattr(S, "movedim") else S


def dst(f: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sine coefficients b_k with f_j = sum_k b_k sin(pi j k / N) (interior grid j = 1..N-1)."""
    N = f.shape[dim] + 1
    return _dst_raw(f, dim) * (2.0 / N)


def idst(b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _dst_raw(b, dim)


def dst2(f):
    return dst(dst(f, -1), -2)


def idst2(b):
    return idst(idst(b, -1), -2)
