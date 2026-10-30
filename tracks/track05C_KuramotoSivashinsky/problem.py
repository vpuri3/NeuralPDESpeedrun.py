"""Track 05C: 1D Kuramoto-Sivashinsky."""
import math

import torch

from speedrun.problem import Problem

NU = 0.01
SIGMA0 = 0.2
T_END = 1.0
NX = 256
NT = 1000

STATEMENT = rf"""
1D Kuramoto-Sivashinsky on x in [-pi, pi), t in [0, {T_END:g}], with periodic boundary conditions in x:

    u_t + u u_x + u_xx + nu u_xxxx = 0,        nu = {NU}
    u(0, x) = exp(-x^2 / (2 sigma0^2)),         sigma0 = {SIGMA0}

Output: u on the prescribed grid x_j = -pi + 2 pi j / {NX}, j = 0..{NX - 1}, at the {NT} snapshot
times t = linspace(0, {T_END:g}, {NT}); shape (1, {NT}, {NX}).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track05C_KuramotoSivashinsky",
        title="1D Kuramoto-Sivashinsky, nu = 0.01",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device),
              "x": -math.pi + 2.0 * math.pi * torch.arange(NX, dtype=dtype, device=device) / NX},
        domain={"x": (-math.pi, math.pi), "t": (0.0, T_END)},
        periodic={"x": True},
        params={"nu": NU},
        fns={"u0": lambda x: torch.exp(-0.5 * (x / SIGMA0) ** 2)},
        device=device,
        dtype=dtype,
    )
