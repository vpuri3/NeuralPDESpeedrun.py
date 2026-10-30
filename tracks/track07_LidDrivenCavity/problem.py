"""Track 07: steady lid-driven cavity, Re = 5000 (benchmark from the PirateNet / SOAP-PINN study)."""
import math

import torch

from speedrun.problem import Problem

RE = 5000.0
N = 256

STATEMENT = rf"""
Steady incompressible Navier-Stokes in the unit square [0, 1]^2, Re = {RE:g} (nu = 1/Re):

    (u . grad) u + grad p = nu Lap u,      div u = 0
    u = 1 - cosh(50 (x - 0.5)) / cosh(25),  v = 0     on the lid y = 1
    u = v = 0                                        on the other three walls

(The lid profile is 1 except within ~0.05 of the corners, which removes the corner singularities.)
Pressure is not scored. L-inf is scored only at points >= 0.05 from the walls.

Output: (u, v) on the {N} x {N} grid x, y = linspace(0, 1, {N}); shape (2, {N}, {N}).
"""


def u_bc(x, y):
    lid = 1.0 - torch.cosh(50.0 * (x - 0.5)) / math.cosh(25.0)
    return torch.where(y > 1.0 - 1e-12, lid, torch.zeros_like(x))


def v_bc(x, y):
    return torch.zeros_like(x)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    g = torch.linspace(0.0, 1.0, N, dtype=dtype, device=device)
    return Problem(
        track="track07_LidDrivenCavity",
        title=f"Lid-driven cavity, Re = {RE:g}",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"x": g, "y": g.clone()},
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        periodic={"x": False, "y": False},
        params={"Re": RE, "nu": 1.0 / RE},
        fns={"u_bc": u_bc, "v_bc": v_bc},
        rel_l2_tol=5e-2,
        linf_tol=1e-1,
        linf_margin=0.05,
        device=device,
        dtype=dtype,
    )
