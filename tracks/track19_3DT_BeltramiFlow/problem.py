"""Track 19: 3D unsteady Beltrami flow (Ethier & Steinman 1994), Re = 1, exact solution."""
import torch

from speedrun.problem import Problem, xp_of

A = D = 1.0
NU = 1.0
NX, T_OUT = 33, (0.25, 0.5, 0.75, 1.0)

STATEMENT = rf"""
3D unsteady incompressible Navier-Stokes on (x, y, z) in [-1, 1]^3, t in [0, 1], nu = {NU:g} (Re = 1):

    u_t + (u . grad) u + grad p = nu Lap u,      div u = 0,        u = (u, v, w)

Initial velocity at t = 0 and Dirichlet velocity on all six faces for t in [0, 1] are those of the
Ethier-Steinman (1994) Beltrami flow with a = d = {A:g} (problem.fns["u0"], ["v0"], ["w0"] and
problem.fns["u_bc"], ["v_bc"], ["w_bc"], functions of (t, x, y, z)); the interior is unknown to the solver.
Pressure is not scored.

Output: (u, v, w) at t = {", ".join(f"{t:g}" for t in T_OUT)} on the {NX}^3 grid x, y, z = linspace(-1, 1, {NX});
shape (3, {len(T_OUT)}, {NX}, {NX}, {NX}).
"""


def exact(t, x, y, z, a=A, d=D, nu=NU):
    """Ethier-Steinman velocity (u, v, w) and pressure p."""
    xp = xp_of(t, x, y, z)
    e = xp.exp(-nu * d**2 * t)
    u = -a * (xp.exp(a * x) * xp.sin(a * y + d * z) + xp.exp(a * z) * xp.cos(a * x + d * y)) * e
    v = -a * (xp.exp(a * y) * xp.sin(a * z + d * x) + xp.exp(a * x) * xp.cos(a * y + d * z)) * e
    w = -a * (xp.exp(a * z) * xp.sin(a * x + d * y) + xp.exp(a * y) * xp.cos(a * z + d * x)) * e
    p = -a**2 / 2 * (xp.exp(2 * a * x) + xp.exp(2 * a * y) + xp.exp(2 * a * z)
                     + 2 * xp.sin(a * x + d * y) * xp.cos(a * z + d * x) * xp.exp(a * (y + z))
                     + 2 * xp.sin(a * y + d * z) * xp.cos(a * x + d * y) * xp.exp(a * (z + x))
                     + 2 * xp.sin(a * z + d * x) * xp.cos(a * y + d * z) * xp.exp(a * (x + y))) \
        * xp.exp(-2 * nu * d**2 * t)
    return u, v, w, p


def _component(k):
    return lambda t, x, y, z: exact(t, x, y, z)[k]


def _initial(k):
    return lambda x, y, z: exact(xp_of(x).zeros_like(x), x, y, z)[k]


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    g = torch.linspace(-1.0, 1.0, NX, dtype=dtype, device=device)
    return Problem(
        track="track19_3DT_BeltramiFlow",
        title="3D Beltrami flow, Re = 1",
        statement=STATEMENT,
        fields=("u", "v", "w"),
        axes={"t": torch.tensor(T_OUT, dtype=dtype, device=device), "x": g, "y": g.clone(), "z": g.clone()},
        domain={"t": (0.0, 1.0), "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        periodic={"x": False, "y": False, "z": False},
        params={"nu": NU, "Re": 1.0 / NU, "a": A, "d": D},
        fns={"u0": _initial(0), "v0": _initial(1), "w0": _initial(2),
             "u_bc": _component(0), "v_bc": _component(1), "w_bc": _component(2)},
        device=device,
        dtype=dtype,
    )
