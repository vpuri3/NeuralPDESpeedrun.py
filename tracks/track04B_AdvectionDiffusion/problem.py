"""Track 04B: steady 2D advection-diffusion at high Peclet number (boundary layers)."""
import math

import torch

from speedrun.problem import Problem

EPS = 1e-3          # diffusivity; Peclet number |b| L / eps ~ 1e3
N_EVAL = 257

STATEMENT = rf"""
Steady advection-diffusion on the unit square [0, 1]^2:

    -eps Lap u + b . grad u = f,      b = (1, 1),   eps = {EPS:g}
    u = 0 on the boundary
    f(x, y) = g(x) + g(y),   g(s) = s - (exp((s - 1)/eps) - exp(-1/eps)) / (1 - exp(-1/eps))

The solution has exponential boundary layers of width ~eps along the outflow walls x = 1 and y = 1.

Output: u on the tensor grid x_i = y_i = (1 - cos(pi i / {N_EVAL - 1})) / 2, i = 0..{N_EVAL - 1}
(Chebyshev-Gauss-Lobatto points, clustered at the walls so the boundary layers are sampled);
shape (1, {N_EVAL}, {N_EVAL}).
"""


def g(s, eps=EPS):
    return s - (torch.exp((s - 1.0) / eps) - math.exp(-1.0 / eps)) / (1.0 - math.exp(-1.0 / eps))


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    xe = 0.5 - 0.5 * torch.cos(math.pi * torch.arange(N_EVAL, dtype=dtype, device=device) / (N_EVAL - 1))
    return Problem(
        track="track04B_AdvectionDiffusion",
        title="2D advection-diffusion, Pe ~ 1e3",
        statement=STATEMENT,
        fields=("u",),
        axes={"x": xe, "y": xe.clone()},
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        periodic={"x": False, "y": False},
        params={"eps": EPS, "b": (1.0, 1.0)},
        fns={"forcing": lambda x, y: g(x) + g(y), "bc": lambda x, y: torch.zeros_like(x)},
        device=device,
        dtype=dtype,
    )
