"""Reference: closed-form solution (cross-checked against Chebyshev collocation)."""
import math

import torch

from speedrun import plotting
from speedrun.numerics import bary_matrix, cheb

from .problem import make_problem


def exact(x):
    pi = math.pi
    return (pi * x * (-x + pi**2 * (2 * x - 3) + 1) - torch.sin(pi * x)) / pi**3


def chebyshev_solve(problem, N=32):
    x, D = cheb(N, 0.0, 1.0, dtype=problem.dtype, device=problem.device)
    A = D @ D @ D
    b = problem.fns["forcing"](x)
    # replace three rows by the boundary conditions
    A[0] = 0.0
    A[0, 0] = 1.0
    b[0] = problem.fns["u_left"]
    A[-1] = 0.0
    A[-1, -1] = 1.0
    b[-1] = problem.fns["u_right"]
    A[-2] = D[-1]
    b[-2] = problem.fns["du_right"]
    u = torch.linalg.solve(A, b)
    return bary_matrix(x, problem.axes["x"]) @ u


def compute(device="cpu"):
    p = make_problem(device=device)
    u = exact(p.axes["x"])[None]
    check = float((chebyshev_solve(p, 32) - u[0]).abs().max())
    return u, {"method": "closed form (NeuralPDE.jl docs)", "check_chebyshev_N32_linf": check}, {}


def plot(problem, u, extras, figdir):
    plotting.line(problem.axes["x"], {"u (reference)": u[0]}, figdir / "reference.png",
                  title="Track 1: u''' = cos(pi x)")
