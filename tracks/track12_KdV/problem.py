"""Track 12: Korteweg-de Vries equation (benchmark from the PirateNet / SOAP-PINN study)."""
import math

import torch

from speedrun.problem import Problem

DELTA = 0.022
NT, NX = 201, 512

STATEMENT = rf"""
Korteweg-de Vries equation on x in [-1, 1] (periodic), t in [0, 1]:

    u_t + u u_x + {DELTA}^2 u_xxx = 0
    u(0, x) = cos(pi x)

The cosine steepens and breaks into a train of solitons.

Output: u at t_k = k / {NT - 1}, k = 1..{NT - 1}, and x_j = -1 + 2 j / {NX - 1}, j = 0..{NX - 1};
shape (1, {NT - 1}, {NX}).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track12_KdV",
        title="Korteweg-de Vries, delta = 0.022",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, 1.0, NT, dtype=dtype, device=device)[1:],
              "x": torch.linspace(-1.0, 1.0, NX, dtype=dtype, device=device)},
        domain={"t": (0.0, 1.0), "x": (-1.0, 1.0)},
        periodic={"x": True},
        params={"delta": DELTA},
        fns={"u0": lambda x: torch.cos(math.pi * x)},
        rel_l2_tol=1e-3,
        linf_tol=1e-2,
        device=device,
        dtype=dtype,
    )
