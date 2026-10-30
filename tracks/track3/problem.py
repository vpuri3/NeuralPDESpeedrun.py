"""Track 3: 2D variable-coefficient (high-contrast) diffusion."""
import torch

from speedrun.problem import Problem

KAPPA = 1e3          # conductivity of the inclusions (background = 1): contrast 1e3
F0 = 18.0            # uniform source strength (sets max|u| ~ 1)
N_EVAL = 256
EDGES = ((2 / 16, 4 / 16), (7 / 16, 9 / 16), (12 / 16, 14 / 16))   # 3 x 3 array of square inclusions

STATEMENT = rf"""
Steady diffusion with a discontinuous, high-contrast coefficient on the unit square [0, 1]^2:

    -div( a(x, y) grad u ) = f,        u = 0 on the boundary
    a = {KAPPA:g} inside 9 square inclusions [s_i, e_i] x [s_j, e_j],  a = 1 elsewhere,
        (s, e) in {{(2/16, 4/16), (7/16, 9/16), (12/16, 14/16)}}
    f = {F0:g}

(Flux continuity a grad u . n across the inclusion edges; u and the normal flux are continuous.)

Output: u at the {N_EVAL} x {N_EVAL} cell centers x_i = y_i = (i + 1/2) / {N_EVAL}; shape (1, {N_EVAL}, {N_EVAL}).
"""


def coefficient(x, y):
    inx = torch.zeros_like(x, dtype=torch.bool)
    iny = torch.zeros_like(y, dtype=torch.bool)
    for s, e in EDGES:
        inx = inx | ((x > s) & (x < e))
        iny = iny | ((y > s) & (y < e))
    return torch.where(inx & iny, torch.full_like(x, KAPPA), torch.ones_like(x))


def forcing(x, y):
    return torch.full_like(x, F0)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    xc = (torch.arange(N_EVAL, dtype=dtype, device=device) + 0.5) / N_EVAL
    return Problem(
        track="track3",
        title="2D high-contrast diffusion (contrast 1e3)",
        statement=STATEMENT,
        fields=("u",),
        axes={"x": xc, "y": xc.clone()},
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={"kappa": KAPPA, "f0": F0, "inclusion_edges": EDGES},
        fns={"a": coefficient, "forcing": forcing, "bc": lambda x, y: torch.zeros_like(x)},
        device=device,
        dtype=dtype,
    )
