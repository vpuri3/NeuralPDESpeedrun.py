"""Track 15: Grey-Scott reaction-diffusion (benchmark from the PirateNet / SOAP-PINN study)."""
import torch

from speedrun.problem import Problem
from tracks.data_common import load_problem_data, periodic_interp

TRACK = "track15_2DT_GreyScott"
B1, B2, C1, C2, EPS1, EPS2 = 40.0, 100.0, 1000.0, 1000.0, 0.2, 0.1

STATEMENT = rf"""
Grey-Scott reaction-diffusion system on (x, y) in [-1, 1]^2 (periodic), t in [0, 2]:

    u_t = eps1 Lap u + b1 (1 - u) - c1 u v^2,      eps1 = {EPS1:g}, b1 = {B1:g}, c1 = {C1:g}
    v_t = eps2 Lap v - b2 v      + c2 u v^2,      eps2 = {EPS2:g}, b2 = {B2:g}, c2 = {C2:g}

Initial condition: (u, v)(0, x, y) given as samples on the output grid (problem.fns["ic"]; problem.fns["u0"],
["v0"] are their trigonometric interpolants).

Output: (u, v) at t_k = 0.02 k, k = 1..100, on the 200 x 200 grid x_i, y_j = -1 + 2 i / 199;
shape (2, 100, 200, 200).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    d = load_problem_data(TRACK, device, dtype)
    lo, L = (-1.0, -1.0), (2.0, 2.0)
    return Problem(
        track=TRACK,
        title="Grey-Scott reaction-diffusion",
        statement=STATEMENT,
        fields=("u", "v"),
        axes={"t": d["t"], "x": d["x"], "y": d["y"]},
        domain={"t": (0.0, 2.0), "x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        periodic={"x": True, "y": True},
        params={"b1": B1, "b2": B2, "c1": C1, "c2": C2, "eps1": EPS1, "eps2": EPS2},
        fns={"u0": periodic_interp(d["u0"], lo, L), "v0": periodic_interp(d["v0"], lo, L),
             "ic": {"x": d["x"], "y": d["y"], "u": d["u0"], "v": d["v0"]}},
        device=device,
        dtype=dtype,
    )
