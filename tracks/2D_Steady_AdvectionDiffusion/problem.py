"""2D_Steady_AdvectionDiffusion: steady 2D advection-diffusion (heat transport by a cellular flow between a cold and a hot wall).

Binous, Bellagi & Higgins, "Steady-State Two-Dimensional Convection-Diffusion Equation", Wolfram Demonstrations
Project (2015; its Mathematica code solves exactly this with NDSolve FEM, the y walls left natural), after Biringen & Chow, "An Introduction to Computational Fluid Mechanics by Example" (2011).
"""
import math

import torch

from speedrun.problem import Problem, xp_of

RE = 40.0           # Reynolds number   (the Demonstration's sliders: Re in [5, 100], Pr in [0.5, 5])
PR = 1.6            # Prandtl number
PE = RE * PR        # Peclet number = 64: only the product enters
N_OUT = 256         # output: uniform grid, h = 1 / N_OUT

STATEMENT = rf"""
Steady advection-diffusion of temperature T(x, y) on [0, 1] x [0, 2] by a prescribed cellular flow:

    (1 / (Re Pr)) (T_xx + T_yy) - u T_x - v T_y = 0,      Re = {RE:g}, Pr = {PR:g}  (Pe = Re Pr = {PE:g})
    u = -sin(pi x) cos(pi y),   v = cos(pi x) sin(pi y)      (two counter-rotating cells, divergence-free)
    T(0, y) = 0 (cold wall),   T(1, y) = 1 (hot wall),   T_y(x, 0) = T_y(x, 2) = 0 (insulated walls)

The flow sweeps hot fluid from the hot wall along the insulated walls toward the cold wall, and cold fluid
back along the cell boundary y = 1, bending the isotherms into an S around each cell.

Output: T on the uniform grid x_i = i / {N_OUT}, i = 0..{N_OUT}, y_j = j / {N_OUT}, j = 0..{2 * N_OUT};
shape (1, {N_OUT + 1}, {2 * N_OUT + 1}).
"""


def velocity(x, y):
    xp = xp_of(x, y)
    return -xp.sin(math.pi * x) * xp.cos(math.pi * y), xp.cos(math.pi * x) * xp.sin(math.pi * y)


def output_axes(dtype=torch.float64, device="cpu"):
    x = torch.arange(N_OUT + 1, dtype=dtype, device=device) / N_OUT
    return x, torch.arange(2 * N_OUT + 1, dtype=dtype, device=device) / N_OUT


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    x, y = output_axes(dtype, device)
    return Problem(
        track="2D_Steady_AdvectionDiffusion",
        title=f"2D steady advection-diffusion, cellular flow, Pe = {PE:g}",
        statement=STATEMENT,
        fields=("T",),
        axes={"x": x, "y": y},
        domain={"x": (0.0, 1.0), "y": (0.0, 2.0)},
        periodic={"x": False, "y": False},
        params={"Re": RE, "Pr": PR, "Pe": PE},
        fns={"velocity": velocity,
             "bc": lambda x, y: xp_of(x).where(x > 0.5, xp_of(x).ones_like(x), xp_of(x).zeros_like(x))},
        device=device,
        dtype=dtype,
    )
