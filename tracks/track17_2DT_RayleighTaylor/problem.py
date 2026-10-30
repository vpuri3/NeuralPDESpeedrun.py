"""Track 17: Rayleigh-Taylor instability, Ra = 1e6 (benchmark from the PirateNet / SOAP-PINN study)."""
import torch

from speedrun.problem import Problem
from tracks.data_common import load_problem_data

TRACK = "track17_2DT_RayleighTaylor"

STATEMENT = r"""
Rayleigh-Taylor instability (Boussinesq), Ra = 1e6, Pr = 0.71, on x in [0, 1], y in [0, 2], t in [0, 8.5]:

    u_t + (u . grad) u + p_x = a1 Lap u
    v_t + (u . grad) v + p_y = a1 Lap v + a2 T
    u_x + v_y = 0
    T_t + (u . grad) T       = a4 Lap T,        a1 = sqrt(Pr / Ra), a2 = 1, a4 = 1 / sqrt(Ra Pr)

Boundary conditions as in the benchmark formulation: u = v = 0 and T = 0 on the walls y = 0 and y = 2; x is
treated as periodic. (The reference data are consistent with symmetric side walls at x = 0, 1.)
Initial condition: (u, v, p, T) at t = 0 given as samples on the output grid (problem.fns["ic"]).
The grid is non-uniform (clustered near the edges).

Output: (u, v, T) at the 34 snapshot times problem.axes["t"] (spacing 0.25) on the 100 x 200 (x, y) grid
problem.axes["x"], ["y"]; shape (3, 34, 100, 200). Pressure is not scored.
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    d = load_problem_data(TRACK, device, dtype)
    return Problem(
        track=TRACK,
        title="Rayleigh-Taylor instability, Ra = 1e6",
        statement=STATEMENT,
        fields=("u", "v", "T"),
        axes={"t": d["t"], "x": d["x"], "y": d["y"]},
        domain={"t": (0.0, float(d["t"][-1])), "x": (0.0, 1.0), "y": (0.0, 2.0)},
        periodic={"x": True, "y": False},
        params={"Ra": float(d["Ra"]), "Pr": float(d["Pr"]), "a1": float(d["a1"]), "a2": float(d["a2"]),
                "a4": float(d["a4"])},
        fns={"ic": {"x": d["x"], "y": d["y"], "u": d["u0"], "v": d["v0"], "p": d["p0"], "T": d["T0"]}},
        rel_l2_tol=2e-2,
        linf_tol=1e-1,
        device=device,
        dtype=dtype,
    )
