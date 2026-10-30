"""Track 05: Allen-Cahn equation (benchmark from the PirateNet / SOAP-PINN study)."""
import math

import torch

from speedrun.problem import Problem, xp_of

EPS, KAPPA = 1e-4, 5.0
NT, NX = 201, 512

STATEMENT = rf"""
Allen-Cahn equation on x in [-1, 1] (periodic), t in [0, 1]:

    u_t - {EPS:g} u_xx + {KAPPA:g} u^3 - {KAPPA:g} u = 0
    u(0, x) = x^2 cos(pi x)

Thin interfaces form between the stable states u = -1 and u = +1.

Output: u at t_k = k / {NT - 1}, k = 1..{NT - 1}, and x_j = -1 + 2 j / {NX - 1}, j = 0..{NX - 1};
shape (1, {NT - 1}, {NX}).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track05_1DT_AllenCahn",
        title="Allen-Cahn, eps = 1e-4",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, 1.0, NT, dtype=dtype, device=device)[1:],
              "x": torch.linspace(-1.0, 1.0, NX, dtype=dtype, device=device)},
        domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
        periodic={"x": True},
        params={"eps": EPS, "kappa": KAPPA},
        fns={"u0": lambda x: x**2 * xp_of(x).cos(math.pi * x)},
        device=device,
        dtype=dtype,
    )
