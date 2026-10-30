"""Track 5A: 1D viscous Burgers (SNF-ROM / NeuralROMs paper, arXiv:2405.14890, sec. 6.3)."""
import math

import torch

from speedrun.problem import Problem

MU = 0.6          # IC parameter (training case shown in Fig. 10(a) of the paper)
NU = 1e-4         # viscosity (paper: nu = 1e-4, Re = 10,000)
T_END = 0.5
NX = 1024         # prescribed collocation grid
NT = 500          # snapshots

STATEMENT = rf"""
1D viscous Burgers on the periodic domain x in [0, 2), t in [0, {T_END}]:

    u_t + u u_x = nu u_xx,          nu = {NU:g}
    u(0, x) = 1 + (mu / 2) (1 + sin(2 pi x - pi / 2))   for x in [0, 1]
            = 1                                         otherwise,          mu = {MU}

(The paper's eq. (58) prints "0 otherwise"; the data-generation code in NeuralROMs.jl uses 1, which
makes the initial condition continuous. We follow the code.)

Output: u on the prescribed grid x_j = 2 j / {NX}, j = 0..{NX - 1}, at the {NT} snapshot times
t = linspace(0, {T_END}, {NT}); shape (1, {NT}, {NX}).
"""


def u0(x, mu=MU):
    inside = (x >= 0.0) & (x <= 1.0)
    bump = 1.0 + 0.5 * mu * (1.0 + torch.sin(2.0 * math.pi * x - 0.5 * math.pi))
    return torch.where(inside, bump, torch.ones_like(x))


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track5A",
        title="1D viscous Burgers, nu = 1e-4",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device),
              "x": 2.0 * torch.arange(NX, dtype=dtype, device=device) / NX},
        domain={"x": (0.0, 2.0), "periodic": True, "t": (0.0, T_END)},
        params={"nu": NU, "mu": MU},
        fns={"u0": lambda x: u0(torch.remainder(x, 2.0))},
        device=device,
        dtype=dtype,
    )
