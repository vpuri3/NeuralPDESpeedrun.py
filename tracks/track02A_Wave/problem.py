"""Track 2A: 1D wave equation with Dirichlet boundary conditions (NeuralPDE.jl wave example #1)."""
import torch

from speedrun.problem import Problem

STATEMENT = r"""
1D wave equation, c = 1, on (t, x) in [0, 1] x [0, 1]:

    u_tt = c^2 u_xx
    u(t, 0) = u(t, 1) = 0
    u(0, x) = x (1 - x),     u_t(0, x) = 0

Output: u on the tensor grid t = linspace(0, 1, 201) x x = linspace(0, 1, 201), shape (1, 201, 201).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track02A_Wave",
        title="1D wave equation (Dirichlet)",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, 1.0, 201, dtype=dtype, device=device),
              "x": torch.linspace(0.0, 1.0, 201, dtype=dtype, device=device)},
        domain={"t": (0.0, 1.0), "x": (0.0, 1.0)},
        params={"c": 1.0},
        fns={
            "u0": lambda x: x * (1.0 - x),
            "ut0": lambda x: torch.zeros_like(x),
            "bc_left": lambda t: torch.zeros_like(t),
            "bc_right": lambda t: torch.zeros_like(t),
        },
        device=device,
        dtype=dtype,
    )
