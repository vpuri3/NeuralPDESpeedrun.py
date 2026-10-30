"""Track 05B: 2D viscous Burgers (vector form)."""
import math

import torch

from speedrun.problem import Problem

MU = 1.0         # IC amplitude (paper's test case)
NU = 1e-3
T_END = 0.5
NX = 512         # prescribed collocation grid (NX x NX)
NT = 11          # evaluated snapshots t = 0, 0.05, ..., 0.5
X0, LX = -0.25, 1.0

STATEMENT = rf"""
2D viscous Burgers (vector form) on the periodic square [-0.25, 0.75)^2, t in [0, {T_END}]:

    u_t + (u . grad) u = nu Lap u,      u = (u, v),   nu = {NU:g}
    u(0, x, y) = v(0, x, y) = mu sin(2 pi x) sin(2 pi y)   for (x, y) in [0, 0.5]^2
               = 0                                          otherwise,           mu = {MU}

A shock forms around t = 0.5.

Output: (u, v) on the prescribed {NX} x {NX} grid x_i = -0.25 + i / {NX}, y_j = -0.25 + j / {NX}, at the
{NT} snapshot times t = linspace(0, {T_END}, {NT}); shape (2, {NT}, {NX}, {NX}).
"""


def u0(x, y, mu=MU):
    inside = (x >= 0.0) & (x <= 0.5) & (y >= 0.0) & (y <= 0.5)
    return torch.where(inside, mu * torch.sin(2 * math.pi * x) * torch.sin(2 * math.pi * y), torch.zeros_like(x))


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    g = X0 + LX * torch.arange(NX, dtype=dtype, device=device) / NX
    return Problem(
        track="track05B_Burgers2D",
        title="2D viscous Burgers, nu = 1e-3",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device), "x": g, "y": g.clone()},
        domain={"x": (X0, X0 + LX), "y": (X0, X0 + LX), "periodic": True, "t": (0.0, T_END)},
        params={"nu": NU, "mu": MU},
        fns={"u0": u0, "v0": u0},
        device=device,
        dtype=dtype,
    )
