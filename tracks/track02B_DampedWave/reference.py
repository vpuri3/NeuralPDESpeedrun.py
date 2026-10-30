"""Reference: exact Fourier-sine series with analytic initial-data coefficients, K = 2^15 modes.

Each sine mode a_k(t) solves a_k'' + beta a_k' + w_k^2 a_k = 0, beta = b v^2, w_k = v k pi, exactly:
    a_k(t) = exp(-beta t / 2) [A_k cos(mu_k t) + B_k sin(mu_k t)],  mu_k = sqrt(w_k^2 - beta^2 / 4)
with sine coefficients of the initial data (exact integrals):
    x(1-x):  8 / (k pi)^3 for odd k,      1 - 2x:  4 / (k pi) for even k.
Truncation error of the series at K = 32768 is O(K^-1.5) ~ 1e-7 in the l2 sense.
"""
import math

import torch

from speedrun import plotting

from .problem import make_problem


def modal(t, x, a0, a1, k, v, b, chunk=4096):
    beta = b * v**2
    w = v * k * math.pi
    mu = torch.sqrt(w**2 - beta**2 / 4)            # all modes underdamped for v = 3, b = 2
    A = a0
    B = (a1 + 0.5 * beta * a0) / mu
    out = torch.zeros(t.numel(), x.numel(), dtype=t.dtype, device=t.device)
    decay = torch.exp(-0.5 * beta * t)[:, None]
    for s in range(0, k.numel(), chunk):
        sl = slice(s, s + chunk)
        amp = decay * (A[sl] * torch.cos(t[:, None] * mu[sl]) + B[sl] * torch.sin(t[:, None] * mu[sl]))
        out = out + amp @ torch.sin(math.pi * x[:, None] * k[None, sl]).T
    return out


def compute(device="cpu"):
    p = make_problem(device=device)
    t, x = p.axes["t"], p.axes["x"]
    v, b = p.params["v"], p.params["b"]
    res = {}
    for K in (2**13, 2**15):
        k = torch.arange(1, K + 1, dtype=t.dtype, device=t.device)
        odd = (torch.remainder(k, 2) == 1).to(t.dtype)
        a0 = odd * 8.0 / (k * math.pi) ** 3
        a1 = (1 - odd) * 4.0 / (k * math.pi)
        res[K] = modal(t, x, a0, a1, k, v, b)
    u = res[2**15][None]
    chk = float((res[2**13] - res[2**15]).abs().max())
    return u, {"method": "exact modal sine series, K=32768", "check_K8192_vs_K32768_linf": chk}, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 02B: damped wave u(t, x)")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", n=7, title="Track 02B snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 2B")
