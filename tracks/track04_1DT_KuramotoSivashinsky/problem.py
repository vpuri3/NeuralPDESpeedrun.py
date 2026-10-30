"""Track 04: chaotic Kuramoto-Sivashinsky (benchmark from the PirateNet / SOAP-PINN study)."""
import math

import torch

from speedrun.problem import Problem
from tracks.data_common import load_problem_data, periodic_interp

TRACK = "track04_1DT_KuramotoSivashinsky"
A, B, C = 100 / 16, 100 / 16**2, 100 / 16**4

STATEMENT = r"""
Kuramoto-Sivashinsky equation on x in [0, 2 pi] (periodic), t in [0, 1], in the chaotic regime:

    u_t + a u u_x + b u_xx + c u_xxxx = 0,      a = 100/16,  b = 100/16^2,  c = 100/16^4

Initial condition: u(0, x) is given as samples on the output x grid (problem.fns["ic"]; problem.fns["u0"] is
its trigonometric interpolant). Small errors grow exponentially; the bar reflects that.

Output: u at t_k = k / 250, k = 1..250, and x_j = 2 pi j / 511, j = 0..511; shape (1, 250, 512).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    d = load_problem_data(TRACK, device, dtype)
    return Problem(
        track=TRACK,
        title="Kuramoto-Sivashinsky (chaotic)",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": d["t"], "x": d["x"]},
        domain={"t": (0.0, 1.0), "x": (0.0, 2 * math.pi)},
        periodic={"x": True},
        params={"a": A, "b": B, "c": C},
        fns={"u0": periodic_interp(d["u0"], 0.0, 2 * math.pi), "ic": {"x": d["x"], "u": d["u0"]}},
        rel_l2_tol=5e-2,
        linf_tol=0.5,
        device=device,
        dtype=dtype,
    )
