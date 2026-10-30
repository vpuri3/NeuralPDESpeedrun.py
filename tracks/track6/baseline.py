"""Record #0 (classical baseline): Chebyshev collocation of the streamfunction form of steady
Navier-Stokes, N x N = 16 x 16, Newton with continuation in viscosity, dense LU."""
from speedrun.numerics.streamfunction import StreamfunctionSolver

N = 16


def solve(problem):
    p = problem
    S = StreamfunctionSolver(N, N, p.domain["x"], p.domain["y"], p.fns["u_bc"], p.fns["v_bc"],
                             dtype=p.dtype, device=p.device)
    S.solve(p.params["nu"], nu_path=[1.0, 0.2])
    return S.evaluate(p.axes["x"], p.axes["y"])
