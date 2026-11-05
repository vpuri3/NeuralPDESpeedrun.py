"""Track 09: steady 2D advection-diffusion (heat transport by a cellular flow between a cold and a hot wall).

Binous, Bellagi & Higgins, "Steady-State Two-Dimensional Convection-Diffusion Equation", Wolfram Demonstrations
Project (2015), after Biringen & Chow, "An Introduction to Computational Fluid Mechanics by Example" (2011).
"""
import math

import torch

from speedrun.numerics import cheb
from speedrun.problem import Problem, xp_of

RE = 100.0          # Reynolds number
PR = 7.0            # Prandtl number (water)
PE = RE * PR        # Peclet number: only the product enters; thermal layers of width ~ Pe^(-1/2) ~ 0.04
N_CGL = 256         # output: CGL points per unit length in x and in each cell in y

STATEMENT = rf"""
Steady advection-diffusion of temperature T(x, y) on [0, 1] x [0, 2] by a prescribed cellular flow:

    (1 / (Re Pr)) (T_xx + T_yy) - u T_x - v T_y = 0,      Re = {RE:g}, Pr = {PR:g}  (Pe = Re Pr = {PE:g})
    u = -sin(pi x) cos(pi y),   v = cos(pi x) sin(pi y)      (two counter-rotating cells, divergence-free)
    T(0, y) = 0 (cold wall),   T(1, y) = 1 (hot wall),   T_y(x, 0) = T_y(x, 2) = 0 (insulated walls)

The flow carries heat around each cell; at this Pe the isotherms wind into spirals, the walls carry thermal
boundary layers of width ~ Pe^(-1/2), and a thin cold tongue runs along the cell boundary y = 1.

Output: T on the tensor grid x_i = (1 - cos(pi i / {N_CGL})) / 2, i = 0..{N_CGL}, and y = the same Chebyshev-Gauss-
Lobatto points on [0, 1] followed by their mirror images 2 - y on (1, 2] (clustered at the walls and at y = 1);
shape (1, {N_CGL + 1}, {2 * N_CGL + 1}).
"""


def velocity(x, y):
    xp = xp_of(x, y)
    return -xp.sin(math.pi * x) * xp.cos(math.pi * y), xp.cos(math.pi * x) * xp.sin(math.pi * y)


def output_axes(dtype=torch.float64, device="cpu"):
    xc = cheb(N_CGL, 0.0, 1.0, dtype=dtype, device=device)[0]          # ascending, 0 .. 1
    return xc, torch.cat([xc, 2.0 - xc.flip(0)[1:]])


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    x, y = output_axes(dtype, device)
    return Problem(
        track="track09_2D_AdvectionDiffusion",
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
