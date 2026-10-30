"""Reference: Kovasznay's closed-form solution (cross-checked with the Chebyshev streamfunction solver)."""
import torch

from speedrun import plotting
from speedrun.numerics.streamfunction import StreamfunctionSolver

from .problem import make_problem, u_bc, v_bc


def compute(device="cpu"):
    p = make_problem(device=device)
    X, Y = p.grid()
    u = torch.stack([u_bc(X, Y), v_bc(X, Y)])
    S = StreamfunctionSolver(24, 24, p.domain["x"], p.domain["y"], u_bc, v_bc, dtype=p.dtype, device=p.device)
    S.solve(p.params["nu"], nu_path=[1.0, 0.2])
    chk = float((S.evaluate(p.axes["x"], p.axes["y"]) - u).abs().max())
    return u, {"method": "closed form (Kovasznay 1948)", "check_chebyshev_N24_linf": chk}, {}


def plot(problem, u, extras, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    plotting.fields2d(x, y, {"u": u[0], "v": u[1]}, figdir / "reference.png",
                      suptitle="Track 6: Kovasznay flow, Re = 20", cmap="viridis", streamplot=True)
