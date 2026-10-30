"""Reference: closed form. The initial velocity is a sum of Laplacian eigenmodes with |k|^2 = 25, so the
nonlinear term is a pure gradient and u(t) = exp(-25 nu t) u(0).
Cross-checked against the Fourier pseudo-spectral baseline at N = 64."""
import torch

from speedrun import plotting

from .problem import make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    T, X, Y = p.grid()
    decay = torch.exp(-25.0 * p.params["nu"] * T)
    u = torch.stack([decay * p.fns["u0"](X, Y), decay * p.fns["v0"](X, Y)])
    from .baseline import fourier_solve
    chk = float((fourier_solve(p, 64) - u).abs().max())
    return u, {"method": "closed form (Walsh 1992)", "check_fourier_N64_linf": chk}, {}


def plot(problem, u, extras, figdir):
    x, y, t = problem.axes["x"], problem.axes["y"], problem.axes["t"]
    for k in (0, len(t) - 1):
        plotting.fields2d(x, y, {"u": u[0, k], "v": u[1, k]}, figdir / f"reference_t{float(t[k]):.1f}.png",
                          suptitle=f"Track 08: decaying eddies, t = {float(t[k]):.1f}", streamplot=True)
