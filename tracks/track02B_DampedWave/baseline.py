"""Record #0 (classical baseline): Fourier-sine (DST-I) spectral method in x with exact modal
time integration of the damped oscillators. Initial data sampled on an N-point grid."""
import math

import torch

from speedrun.numerics import dst

N = 512


def solve(problem):
    dt, dev = problem.dtype, problem.device
    v, b = problem.params["v"], problem.params["b"]
    t, xe = problem.axes["t"], problem.axes["x"]
    xg = torch.arange(1, N, dtype=dt, device=dev) / N
    a0 = dst(problem.fns["u0"](xg))
    a1 = dst(problem.fns["ut0"](xg))
    k = torch.arange(1, N, dtype=dt, device=dev)
    beta = b * v**2
    mu = torch.sqrt((v * k * math.pi) ** 2 - beta**2 / 4)
    B = (a1 + 0.5 * beta * a0) / mu
    amp = torch.exp(-0.5 * beta * t)[:, None] * (a0 * torch.cos(t[:, None] * mu) + B * torch.sin(t[:, None] * mu))
    return (amp @ torch.sin(math.pi * xe[:, None] * k[None, :]).T)[None]
