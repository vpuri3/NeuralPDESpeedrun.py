"""Reference: exact sine series. The Gaussian pulse vanishes at the walls to ~1e-49, so its sine coefficients
are the closed-form Gaussian integrals a_k = 2 sigma sqrt(2 pi) exp(-(k pi sigma)^2 / 2) sin(k pi x0). Each
mode evolves exactly: a_k(t) = exp(-gamma t / 2) a_k [cos(m_k t) + gamma / (2 m_k) sin(m_k t)],
m_k = sqrt((c k pi)^2 - gamma^2 / 4). K = 512 modes (coefficients < 1e-120 beyond); checked against K = 256."""
import math

import torch

from speedrun import plotting

from .baseline import modal
from .problem import SIGMA, X0, make_problem


def exact_coefficients(K, dtype, device):
    k = torch.arange(1, K + 1, dtype=dtype, device=device)
    gauss = torch.exp(-((k * math.pi * SIGMA) ** 2) / 2)
    return 2 * SIGMA * math.sqrt(2 * math.pi) * gauss * torch.sin(k * math.pi * X0)


def compute(device="cpu"):
    p = make_problem(device=device)
    t, x = p.axes["t"], p.axes["x"]
    c, g = p.params["c"], p.params["gamma"]
    u = modal(t, x, exact_coefficients(512, t.dtype, t.device), c, g)
    chk = modal(t, x, exact_coefficients(256, t.dtype, t.device), c, g)
    return u[None], {"method": "exact modal sine series, closed-form coefficients, K=512",
                     "check_K256_linf": float((chk - u).abs().max())}, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 02: damped wave, narrow pulse")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", n=7, title="Track 02 snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 02")
