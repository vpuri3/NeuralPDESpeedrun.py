"""Track 1: third-order linear ODE (NeuralPDE.jl example "ODE with a 3rd-order derivative")."""
import math

import torch

from speedrun.problem import Problem

STATEMENT = r"""
Third-order boundary value problem on x in [0, 1]:

    u'''(x) = cos(pi x)
    u(0) = 0,   u(1) = cos(pi) = -1,   u'(1) = 1

(The NeuralPDE.jl page writes u'(0) = 1 in the text but imposes Dx(u(1)) = 1 in code and in its
analytic solution; we follow the code: u'(1) = 1.)

Output: u at the 1001 uniformly spaced points x = linspace(0, 1, 1001).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track1",
        title="3rd-order ODE  u''' = cos(pi x)",
        statement=STATEMENT,
        fields=("u",),
        axes={"x": torch.linspace(0.0, 1.0, 1001, dtype=dtype, device=device)},
        domain={"x": (0.0, 1.0)},
        params={},
        fns={
            "forcing": lambda x: torch.cos(math.pi * x),     # right-hand side of u''' = f
            "u_left": 0.0,                                     # u(0)
            "u_right": math.cos(math.pi),                      # u(1)
            "du_right": 1.0,                                   # u'(1)
        },
        device=device,
        dtype=dtype,
    )
