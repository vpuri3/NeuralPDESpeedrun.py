"""Chebyshev collocation utilities (Chebyshev-Gauss-Lobatto points, ascending order)."""
from __future__ import annotations

import math

import torch


def cheb(N: int, a: float = -1.0, b: float = 1.0, dtype=torch.float64, device="cpu"):
    """Chebyshev-Gauss-Lobatto nodes on [a, b] (ascending) and the first-derivative matrix.

    Returns x (N+1,), D (N+1, N+1) such that (D @ f(x)) ~= f'(x).  (Trefethen, Spectral Methods in MATLAB)
    """
    j = torch.arange(N + 1, dtype=dtype, device=device)
    x = torch.cos(math.pi * j / N)                                    # descending on [-1, 1]
    c = torch.ones(N + 1, dtype=dtype, device=device)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** j
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + torch.eye(N + 1, dtype=dtype, device=device))
    D = D - torch.diag(D.sum(dim=1))
    # ascending order on [a, b]
    x = torch.flip(x, dims=[0])
    D = torch.flip(D, dims=[0, 1])
    x = a + (b - a) * (x + 1.0) / 2.0
    D = D * (2.0 / (b - a))
    return x, D


def bary_matrix(xn: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
    """Barycentric interpolation matrix P (len(xe), len(xn)) from Chebyshev-Lobatto nodes xn to points xe."""
    n = xn.numel()
    w = torch.ones(n, dtype=xn.dtype, device=xn.device)
    w[0] = 0.5
    w[-1] = 0.5
    w = w * (-1.0) ** torch.arange(n, dtype=xn.dtype, device=xn.device)
    d = xe[:, None] - xn[None, :]
    exact = d.abs() < 1e-14
    d = torch.where(exact, torch.ones_like(d), d)
    P = w[None, :] / d
    P = P / P.sum(dim=1, keepdim=True)
    hit = exact.any(dim=1)
    P = torch.where(hit[:, None], exact.to(P.dtype), P)
    return P


def clenshaw_curtis_weights(N: int, a: float = -1.0, b: float = 1.0, dtype=torch.float64, device="cpu"):
    """Clenshaw-Curtis quadrature weights for the ascending Lobatto nodes of `cheb(N, a, b)`."""
    theta = math.pi * torch.arange(N + 1, dtype=dtype, device=device) / N
    w = torch.zeros(N + 1, dtype=dtype, device=device)
    v = torch.ones(N - 1, dtype=dtype, device=device)
    ii = slice(1, N)
    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1)
        w[N] = w[0]
        for k in range(1, N // 2):
            v = v - 2.0 * torch.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        v = v - torch.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = 1.0 / N**2
        w[N] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v = v - 2.0 * torch.cos(2 * k * theta[ii]) / (4 * k * k - 1)
    w[ii] = 2.0 * v / N
    return torch.flip(w, dims=[0]) * (b - a) / 2.0
