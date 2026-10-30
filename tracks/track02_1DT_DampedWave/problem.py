"""Track 02: 1D damped wave equation, a narrow pulse bouncing between Dirichlet walls."""

import torch

from speedrun.problem import Problem, xp_of

C = 1.0          # wave speed
GAMMA = 0.5      # damping
X0, SIGMA = 0.3, 0.02
T_END = 4.0
NT, NX = 801, 1001

STATEMENT = rf"""
1D damped wave equation on (t, x) in [0, {T_END:g}] x [0, 1]:

    u_tt = c^2 u_xx - gamma u_t,        c = {C:g},  gamma = {GAMMA:g}
    u(t, 0) = u(t, 1) = 0
    u(0, x) = exp(-(x - {X0})^2 / (2 sigma^2)),  sigma = {SIGMA},     u_t(0, x) = 0

The pulse splits into two halves that reflect (with sign flips) off the walls; over [0, {T_END:g}] each crosses
the domain {T_END * C:g} times while decaying like exp(-gamma t / 2).

Output: u on the tensor grid t = linspace(0, {T_END:g}, {NT}) x x = linspace(0, 1, {NX}); shape (1, {NT}, {NX}).
"""


def u0(x):
    return xp_of(x).exp(-((x - X0) ** 2) / (2 * SIGMA**2))


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track02_1DT_DampedWave",
        title="1D damped wave, narrow pulse",
        statement=STATEMENT,
        fields=("u",),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device),
              "x": torch.linspace(0.0, 1.0, NX, dtype=dtype, device=device)},
        domain={"t": (0.0, T_END), "x": (0.0, 1.0)},
        periodic={"x": False},
        params={"c": C, "gamma": GAMMA},
        fns={
            "u0": u0,
            "ut0": lambda x: xp_of(x).zeros_like(x),
            "bc_left": lambda t: xp_of(t).zeros_like(t),
            "bc_right": lambda t: xp_of(t).zeros_like(t),
        },
        device=device,
        dtype=dtype,
    )

