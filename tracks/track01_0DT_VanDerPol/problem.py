"""Track 01: stiff ODE, the Van der Pol oscillator in the relaxation regime (mu = 1000)."""
import torch

from speedrun.problem import Problem

MU = 1000.0
T_END = 3000.0
NT = 3001

STATEMENT = rf"""
Van der Pol oscillator, t in [0, {T_END:g}], mu = {MU:g} (stiff relaxation oscillations):

    y'' - mu (1 - y^2) y' + y = 0
    y(0) = 2,   y'(0) = 0

The solution alternates between slow drifts (time scale ~ mu) and jumps of size ~3 lasting ~1/mu; the
period is ~ (3 - 2 ln 2) mu ~ 1614, so [0, {T_END:g}] covers almost two periods.

Output: y at the {NT} times t = linspace(0, {T_END:g}, {NT}); shape (1, {NT}).
"""


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    return Problem(
        track="track01_0DT_VanDerPol",
        title=f"Van der Pol oscillator, mu = {MU:g} (stiff)",
        statement=STATEMENT,
        fields=("y",),
        axes={"t": torch.linspace(0.0, T_END, NT, dtype=dtype, device=device)},
        domain={"t": (0.0, T_END)},
        params={"mu": MU},
        fns={"y0": 2.0, "dy0": 0.0},
        device=device,
        dtype=dtype,
    )
