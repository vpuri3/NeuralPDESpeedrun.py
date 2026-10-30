"""Track 6B: Kovasznay flow, Re = 40, in a trapezoid with a parabolic top (spec `kov` case, geom/para.m)."""
import torch

from speedrun.problem import Problem
from tracks.kovasznay_common import EQUATIONS, RE, u_bc, v_bc

NR, NS = 201, 201

STATEMENT = rf"""
Kovasznay flow in a curved quadrilateral: the image of the reference square (r, s) in [-1, 1]^2 under the
Gordon-Hall map of its four edges
    bottom (s = -1): (x, y) = (r/2, -1/2)           top (s = +1):   (x, y) = (r, 1 - r^2/2)
    left   (r = -1): (x, y) = (-(3+s)/4, s/2)       right (r = +1): (x, y) = ((3+s)/4, s/2)
i.e.  x = r (3 + s) / 4,   y = (1 + s)/2 (1 - r^2/2) - (1 - s)/4      (problem.fns["map"]).
A trapezoid with corners (-1/2, -1/2), (1/2, -1/2), (1, 1/2), (-1, 1/2) and a parabolic top edge.

{EQUATIONS}

Output: (u, v) at the mapped points map(r_i, s_j), r = s = linspace(-1, 1, {NR}); shape (2, {NR}, {NS}).
"""


def xmap(r, s):
    return r * (3.0 + s) / 4.0, 0.5 * (1.0 + s) * (1.0 - 0.5 * r * r) - 0.25 * (1.0 - s)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track06B_KovasznayCurved",
        title=f"Kovasznay flow, Re = {RE:g}, curved trapezoid",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"r": torch.linspace(-1.0, 1.0, NR, dtype=dtype, device=device),
              "s": torch.linspace(-1.0, 1.0, NS, dtype=dtype, device=device)},
        domain={"r": (-1.0, 1.0), "s": (-1.0, 1.0), "map": "x = r (3 + s) / 4, y = (1 + s)/2 (1 - r^2/2) - (1 - s)/4"},
        params={"Re": RE, "nu": 1.0 / RE},
        fns={"u_bc": u_bc, "v_bc": v_bc, "map": xmap},
        device=device,
        dtype=dtype,
    )
