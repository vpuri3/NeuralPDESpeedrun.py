"""Track 2B: 1D damped wave equation (NeuralPDE.jl wave example #2)."""
import torch

from speedrun.problem import Problem

STATEMENT = r"""
1D damped wave equation on (t, x) in [0, 1] x [0, 1], with v = 3, b = 2 (as implemented in the
NeuralPDE.jl example code):

    u_xx = (1 / v^2) u_tt + b u_t          i.e.   u_tt = v^2 u_xx - b v^2 u_t
    u(t, 0) = u(t, 1) = 0
    u(0, x) = x (1 - x),     u_t(0, x) = 1 - 2x

Note: u_t(0, x) = 1 - 2x does not vanish at the boundary (incompatible with the Dirichlet data), so
u_t is discontinuous along the characteristics x = +-v t leaving the corners. This is part of the problem.

Output: u on the tensor grid t = linspace(0, 1, 201) x x = linspace(0, 1, 201), shape (1, 201, 201).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track02B_DampedWave",
        title="1D damped wave equation",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, 1.0, 201, dtype=dtype, device=device),
              "x": torch.linspace(0.0, 1.0, 201, dtype=dtype, device=device)},
        domain={"t": (0.0, 1.0), "x": (0.0, 1.0)},
        params={"v": 3.0, "b": 2.0},
        fns={
            "u0": lambda x: x * (1.0 - x),
            "ut0": lambda x: 1.0 - 2.0 * x,
            "bc_left": lambda t: torch.zeros_like(t),
            "bc_right": lambda t: torch.zeros_like(t),
        },
        device=device,
        dtype=dtype,
    )
