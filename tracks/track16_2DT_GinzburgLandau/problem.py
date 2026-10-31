"""Track 16: complex Ginzburg-Landau equation (benchmark from the PirateNet / SOAP-PINN study)."""
import torch

from speedrun.problem import Problem
from tracks.data_common import load_problem_data, periodic_interp

TRACK = "track16_2DT_GinzburgLandau"
EPS, K = 0.004, 10.0

STATEMENT = rf"""
Complex Ginzburg-Landau equation, written for A = u + i v, on (x, y) in [-1, 1]^2 (periodic), t in [0, 1]:

    u_t = eps Lap u + k ( u - (u^2 + v^2) u + 1.5 (u^2 + v^2) v )
    v_t = eps Lap v + k ( v - (u^2 + v^2) v - 1.5 (u^2 + v^2) u ),        eps = {EPS:g}, k = {K:g}

Initial condition: (u, v)(0, x, y) given as samples on the output grid (problem.fns["ic"]; problem.fns["u0"],
["v0"] are their trigonometric interpolants).

Output: (u, v) at t_k = 0.01 k, k = 1..100, on the 200 x 200 grid x_i, y_j = -1 + 2 i / 199;
shape (2, 100, 200, 200).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    d = load_problem_data(TRACK, device, dtype)
    lo, L = (-1.0, -1.0), (2.0, 2.0)
    return Problem(
        track=TRACK,
        title="Ginzburg-Landau",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"t": d["t"], "x": d["x"], "y": d["y"]},
        domain={"t": (0.0, 1.0), "x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        periodic={"x": True, "y": True},
        params={"eps": EPS, "k": K},
        fns={"u0": periodic_interp(d["u0"], lo, L), "v0": periodic_interp(d["v0"], lo, L),
             "ic": {"x": d["x"], "y": d["y"], "u": d["u0"], "v": d["v0"]}},
        device=device,
        dtype=dtype,
    )
