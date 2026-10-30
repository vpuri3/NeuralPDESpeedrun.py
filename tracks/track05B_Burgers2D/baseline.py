"""Record #0 (classical baseline): Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 on a 1024 x 1024 grid
(2x the prescribed grid), dt = 1e-3 (SUBSTEPS = 50 steps per 0.05 snapshot interval), injected onto the
prescribed 512 x 512 grid. The prescribed grid itself is too coarse for the shock (512^2: L-inf 1.6e-3)."""
import torch

from speedrun.numerics.periodic import burgers_2d

SUBSTEPS = 50
R = 2


def solve(problem, solver=None):
    x0, x1 = problem.domain["x"]
    n = R * problem.axes["x"].numel()
    g = x0 + (x1 - x0) * torch.arange(n, dtype=problem.dtype, device=problem.device) / n
    X, Y = torch.meshgrid(g, g, indexing="ij")
    out = burgers_2d(problem.fns["u0"](X, Y), problem.fns["v0"](X, Y), x1 - x0, n, problem.params["nu"],
                     problem.axes["t"], SUBSTEPS, dtype=problem.dtype, device=problem.device, stride=R)
    return out.permute(1, 0, 2, 3)          # (t, field, x, y) -> (field, t, x, y)
