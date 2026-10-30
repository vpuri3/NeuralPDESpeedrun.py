"""Reference: exact d'Alembert solution u = [F(x - ct) + F(x + ct)] / 2, F = odd 2-periodic extension of x(1-x).
Cross-checked against the Fourier sine series of the NeuralPDE.jl docs."""
import math

import torch

from speedrun import plotting

from .problem import make_problem


def F(s):
    r = torch.remainder(s, 2.0)
    return torch.where(r <= 1.0, r * (1.0 - r), -(r - 1.0) * (2.0 - r))


def exact(t, x, c=1.0):
    T, X = torch.meshgrid(t, x, indexing="ij")
    return 0.5 * (F(X - c * T) + F(X + c * T))


def series(t, x, K=4001):
    k = torch.arange(1, K + 1, 2, dtype=t.dtype, device=t.device)
    S = torch.sin(math.pi * x[:, None] * k[None, :]) * (8.0 / (k**3 * math.pi**3))
    C = torch.cos(math.pi * t[:, None] * k[None, :])
    return C @ S.T


def compute(device="cpu"):
    p = make_problem(device=device)
    t, x = p.axes["t"], p.axes["x"]
    u = exact(t, x, p.params["c"])[None]
    chk = float((series(t, x) - u[0]).abs().max())
    return u, {"method": "closed form (d'Alembert)", "check_vs_sine_series_K4001_linf": chk}, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 2A: wave equation u(t, x)")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", title="Track 2A snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 2A")
