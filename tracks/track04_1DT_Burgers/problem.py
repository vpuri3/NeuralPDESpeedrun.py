"""Track 04: 1D viscous Burgers, the Raissi et al. (2019) PINN benchmark with a shock at x = 0."""
import math

import torch

from speedrun.problem import Problem, xp_of

NU = 0.01 / math.pi   # viscosity
T_END = 1.0
NT, NX = 1001, 1001   # output grid: dt = 1e-3, dx = 2e-3

STATEMENT = rf"""
1D viscous Burgers on (t, x) in [0, {T_END:g}] x [-1, 1] (Raissi et al. 2019; Shahab et al. 2026, sec. 3.1):

    u_t + u u_x = nu u_xx,          nu = 0.01 / pi
    u(0, x) = -sin(pi x)
    u(t, -1) = u(t, 1) = 0

The two halves of the sine steepen into a stationary shock at x = 0 (width ~ nu) from t ~ 1 / pi on.

Output: u on the tensor grid t = linspace(0, {T_END:g}, {NT}) x x = linspace(-1, 1, {NX}); shape (1, {NT}, {NX}).
"""


def u0(x):
    return -xp_of(x).sin(math.pi * x)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track04_1DT_Burgers",
        title="1D viscous Burgers, nu = 0.01/pi (Raissi)",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device),
              "x": torch.linspace(-1.0, 1.0, NX, dtype=dtype, device=device)},
        domain={"t": (0.0, T_END), "x": (-1.0, 1.0)},
        periodic={"x": False},
        params={"nu": NU},
        fns={
            "u0": u0,
            "bc_left": lambda t: xp_of(t).zeros_like(t),
            "bc_right": lambda t: xp_of(t).zeros_like(t),
        },
        device=device,
        dtype=dtype,
    )
