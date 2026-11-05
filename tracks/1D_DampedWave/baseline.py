"""Record #0 (classical baseline): Fourier-sine (DST-I) coefficients of the initial pulse sampled on an
N = 256 grid, exact modal time integration of the damped oscillators, synthesis at the output points."""
import math

import torch

from speedrun.numerics import dst

N = 256


def modal(t, x, a, c, gamma):
    k = torch.arange(1, a.numel() + 1, dtype=t.dtype, device=t.device)
    m = torch.sqrt((c * k * math.pi) ** 2 - gamma**2 / 4)
    amp = torch.exp(-0.5 * gamma * t)[:, None] * a * (torch.cos(t[:, None] * m)
                                                      + gamma / (2 * m) * torch.sin(t[:, None] * m))
    return amp @ torch.sin(math.pi * x[:, None] * k[None, :]).T


def solve(problem, state=None):
    dt, dev = problem.dtype, problem.device
    xg = torch.arange(1, N, dtype=dt, device=dev) / N
    a = dst(problem.fns["u0"](xg))
    return modal(problem.axes["t"], problem.axes["x"], a, problem.params["c"], problem.params["gamma"])[None]
