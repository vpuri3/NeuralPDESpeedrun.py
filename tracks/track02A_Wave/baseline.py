"""Record #0 (classical baseline): Fourier-sine (DST-I) spectral method in x with exact
modal time integration. The initial data are sampled on an N-point grid; nothing else is used."""
import math

import torch

from speedrun.numerics import dst

N = 32


def solve(problem, solver=None):
    dt, dev = problem.dtype, problem.device
    c = problem.params["c"]
    t, xe = problem.axes["t"], problem.axes["x"]
    xg = torch.arange(1, N, dtype=dt, device=dev) / N                 # interior grid
    a0 = dst(problem.fns["u0"](xg))                                     # sine coefficients, k = 1..N-1
    a1 = dst(problem.fns["ut0"](xg))
    k = torch.arange(1, N, dtype=dt, device=dev)
    w = c * math.pi * k
    amp = torch.cos(t[:, None] * w) * a0 + torch.sin(t[:, None] * w) * (a1 / w)    # (Nt, N-1)
    S = torch.sin(math.pi * xe[:, None] * k[None, :])                               # (Nx, N-1)
    return (amp @ S.T)[None]
