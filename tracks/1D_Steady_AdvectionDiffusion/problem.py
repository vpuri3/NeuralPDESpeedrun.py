"""1D_Steady_AdvectionDiffusion: steady 1D advection-diffusion with an outflow boundary layer (Chebfun example "Boundary layer for
advection-diffusion equation", L. N. Trefethen, 2010: https://www.chebfun.org/examples/ode-linear/BoundaryLayer.html)."""
import math

import torch

from speedrun.problem import Problem, xp_of

EPS = 1e-2          # diffusivity: Pe = 100, boundary layer of width ~eps at the outflow wall x = 0
N_OUT = 1001        # uniform output points (about 46 inside the layer)

STATEMENT = rf"""
Steady advection-diffusion on [0, 1] (flow toward x = 0):

    -eps u'' - u' = 1,      eps = {EPS:g}
    u(0) = u(1) = 0

Away from x = 0 the solution follows the outer solution 1 - x; it drops to 0 across a boundary layer of width
~eps at the outflow wall x = 0 (u = 1/2 at x ~ eps ln 2, slope u'(0) ~ 1/eps).

Output: u at x_i = i / {N_OUT - 1}, i = 0..{N_OUT - 1}; shape (1, {N_OUT}).
"""


def exact(x, eps=EPS):
    xp = xp_of(x)
    return -xp.expm1(-x / eps) / -math.expm1(-1.0 / eps) - x


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    x = torch.linspace(0.0, 1.0, N_OUT, dtype=dtype, device=device)
    return Problem(
        track="1D_Steady_AdvectionDiffusion",
        title=f"1D steady advection-diffusion, boundary layer, eps = {EPS:g}",
        statement=STATEMENT,
        fields=("u",),
        axes={"x": x},
        domain={"x": (0.0, 1.0)},
        periodic={"x": False},
        params={"eps": EPS, "b": -1.0},
        fns={"forcing": lambda x: xp_of(x).ones_like(x), "bc": lambda x: xp_of(x).zeros_like(x)},
        device=device,
        dtype=dtype,
    )
