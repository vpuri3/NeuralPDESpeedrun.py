"""Track 18: Kolmogorov flow, Re = 1e4 (benchmark from the PirateNet / SOAP-PINN study)."""
import math

import torch

from speedrun.problem import Problem, xp_of
from tracks.data_common import load_problem_data, periodic_interp

TRACK = "track18_2DT_KolmogorovFlow"
NU = 1e-4

STATEMENT = r"""
Forced 2D incompressible Navier-Stokes (Kolmogorov flow), Re = 1e4, on [0, 1)^2 (periodic), t in [0, 1.96]:

    u_t + (u . grad) u + p_x = nu Lap u + 2 sin(4 pi y)
    v_t + (u . grad) v + p_y = nu Lap v,              u_x + v_y = 0,      nu = 1e-4

Initial condition: (u, v)(0, x, y) given as samples on the output grid (problem.fns["ic"]; problem.fns["u0"],
["v0"] are their trigonometric interpolants). The flow is turbulent; the bar reflects that.

Output: (u, v) at t_k = 0.04 k, k = 1..49, on the 256 x 256 grid of cell centers x_i, y_j = (i + 1/2) / 256;
shape (2, 49, 256, 256). Pressure is not scored.
"""


def forcing(x, y):
    xp = xp_of(x, y)
    return 2.0 * xp.sin(4 * math.pi * y), xp.zeros_like(y)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    d = load_problem_data(TRACK, device, dtype)
    lo = (float(d["x"][0]), float(d["y"][0]))
    return Problem(
        track=TRACK,
        title="Kolmogorov flow, Re = 1e4",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"t": d["t"], "x": d["x"], "y": d["y"]},
        domain={"t": (0.0, float(d["t"][-1])), "x": (0.0, 1.0), "y": (0.0, 1.0)},
        periodic={"x": True, "y": True},
        params={"nu": NU, "Re": 1.0 / NU},
        fns={"forcing": forcing,
             "u0": periodic_interp(d["u0"], lo, (1.0, 1.0), endpoint=False),
             "v0": periodic_interp(d["v0"], lo, (1.0, 1.0), endpoint=False),
             "ic": {"x": d["x"], "y": d["y"], "u": d["u0"], "v": d["v0"]}},
        rel_l2_tol=5e-2,
        linf_tol=5e-1,
        device=device,
        dtype=dtype,
    )
