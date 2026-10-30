"""Track 13: decaying eddies (Walsh 1992)."""
import math

import torch

from speedrun.problem import Problem, xp_of

NU = 5e-2
T_END = 5.0
NX = 128
NT = 11
L = 2 * math.pi

STATEMENT = rf"""
Unsteady incompressible Navier-Stokes on the periodic square [0, 2 pi)^2, t in [0, {T_END:g}], nu = {NU:g}:

    u_t + (u . grad) u + grad p = nu Lap u,      div u = 0
    u(0, x, y) = sin(5 y) + cos(3 x) cos(4 y)
    v(0, x, y) = cos(5 x) + 0.75 sin(3 x) sin(4 y)

(Walsh's decaying eddies.) Pressure is NOT scored.

Output: (u, v) on the {NX} x {NX} grid x_i = y_i = 2 pi i / {NX} at the {NT} times t = linspace(0, {T_END:g}, {NT});
shape (2, {NT}, {NX}, {NX}).
"""


def u0(x, y):
    xp = xp_of(x, y)
    return xp.sin(5 * y) + xp.cos(3 * x) * xp.cos(4 * y)


def v0(x, y):
    xp = xp_of(x, y)
    return xp.cos(5 * x) + 0.75 * xp.sin(3 * x) * xp.sin(4 * y)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    g = L * torch.arange(NX, dtype=dtype, device=device) / NX
    return Problem(
        track="track13_2DT_DecayingEddies",
        title=f"Decaying eddies (Walsh), nu = {NU:g}",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device), "x": g, "y": g.clone()},
        domain={"x": (0.0, L), "y": (0.0, L), "t": (0.0, T_END)},
        periodic={"x": True, "y": True},
        params={"nu": NU},
        fns={"u0": u0, "v0": v0},
        device=device,
        dtype=dtype,
    )
