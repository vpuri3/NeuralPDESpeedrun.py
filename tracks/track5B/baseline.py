"""Record #0 (classical baseline): Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 directly on the
prescribed 512 x 512 grid, dt = 1e-3 (SUBSTEPS = 50 steps per 0.05 snapshot interval)."""
import torch

from speedrun.numerics.periodic import burgers_2d

SUBSTEPS = 50


def solve(problem):
    x, y = problem.axes["x"], problem.axes["y"]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    out = burgers_2d(problem.fns["u0"](X, Y), problem.fns["v0"](X, Y), 1.0, x.numel(), problem.params["nu"],
                     problem.axes["t"], SUBSTEPS, dtype=problem.dtype, device=problem.device)
    return out.permute(1, 0, 2, 3)          # (t, field, x, y) -> (field, t, x, y)
