"""Record #0 (classical baseline): the spec's discretization (single-element Legendre PN-PN-2, GLL
quadrature, 3/2 dealiasing), N = 16, steady system solved by Newton with dense LU."""
import torch

from speedrun.numerics.sem import SEMNavierStokes

N = 16


def _solve(p, N):
    S = SEMNavierStokes(N, p.fns["map"], dtype=p.dtype, device=p.device)
    u, v = S.steady(p.params["nu"], p.fns["u_bc"], p.fns["v_bc"])
    (x0, x1), (y0, y1) = p.domain["x"], p.domain["y"]
    return S.interp_rs(torch.stack([u, v]), 2 * (p.axes["x"] - x0) / (x1 - x0) - 1,
                       2 * (p.axes["y"] - y0) / (y1 - y0) - 1)


def warmup(problem):
    _solve(problem, 6)      # initializes torch.func (jacfwd/vmap); the N = 6 result is discarded


def solve(problem):
    return _solve(problem, N)
