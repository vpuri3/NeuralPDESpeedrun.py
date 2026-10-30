"""Track 6A: Kovasznay flow, Re = 40, on the rectangle [-0.5, 2] x [-0.5, 1.5] (spec `kov` case)."""
import torch

from speedrun.numerics.sem import rectangle_map
from speedrun.problem import Problem
from tracks.kovasznay_common import EQUATIONS, RE, u_bc, v_bc

X0, X1, Y0, Y1 = -0.5, 2.0, -0.5, 1.5
NX, NY = 251, 201

STATEMENT = rf"""
Kovasznay flow on the rectangle [{X0}, {X1}] x [{Y0}, {Y1}].

{EQUATIONS}

Output: (u, v) on the tensor grid x = linspace({X0}, {X1}, {NX}) x y = linspace({Y0}, {Y1}, {NY});
shape (2, {NX}, {NY}).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track6A",
        title=f"Kovasznay flow, Re = {RE:g}, rectangle",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"x": torch.linspace(X0, X1, NX, dtype=dtype, device=device),
              "y": torch.linspace(Y0, Y1, NY, dtype=dtype, device=device)},
        domain={"x": (X0, X1), "y": (Y0, Y1)},
        params={"Re": RE, "nu": 1.0 / RE},
        fns={"u_bc": u_bc, "v_bc": v_bc, "map": rectangle_map(X0, X1, Y0, Y1)},
        device=device,
        dtype=dtype,
    )
