"""Track 6: Kovasznay flow (steady incompressible Navier-Stokes with a closed-form solution)."""
import math

import torch

from speedrun.problem import Problem

RE = 20.0
LAM = RE / 2 - math.sqrt(RE**2 / 4 + 4 * math.pi**2)

STATEMENT = rf"""
Steady incompressible Navier-Stokes on the rectangle [-0.5, 1.0] x [-0.5, 1.5], Re = {RE:g}:

    (u . grad) u + grad p = (1/Re) Lap u,      div u = 0

Dirichlet velocity on the whole boundary:
    u = 1 - exp(lam x) cos(2 pi y),   v = (lam / (2 pi)) exp(lam x) sin(2 pi y),
    lam = Re/2 - sqrt(Re^2/4 + 4 pi^2)

(This boundary data is that of the Kovasznay solution; the interior is unknown to the solver.)
Pressure is determined up to a constant and is NOT scored.

Output: (u, v) on the tensor grid x = linspace(-0.5, 1.0, 151) x y = linspace(-0.5, 1.5, 201);
shape (2, 151, 201).
"""


def u_bc(x, y):
    return 1.0 - torch.exp(LAM * x) * torch.cos(2 * math.pi * y)


def v_bc(x, y):
    return LAM / (2 * math.pi) * torch.exp(LAM * x) * torch.sin(2 * math.pi * y)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track6",
        title="Kovasznay flow, Re = 20",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"x": torch.linspace(-0.5, 1.0, 151, dtype=dtype, device=device),
              "y": torch.linspace(-0.5, 1.5, 201, dtype=dtype, device=device)},
        domain={"x": (-0.5, 1.0), "y": (-0.5, 1.5)},
        params={"Re": RE, "nu": 1.0 / RE},
        fns={"u_bc": u_bc, "v_bc": v_bc},
        device=device,
        dtype=dtype,
    )
