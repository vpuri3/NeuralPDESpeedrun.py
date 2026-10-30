"""Reference: the closed-form Ethier-Steinman solution at the output points (exact to round-off; it satisfies
the Navier-Stokes equations to ~1e-15, checked with autograd)."""
import torch

from speedrun import plotting

from .problem import exact, make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    T, X, Y, Z = p.grid()
    u, v, w, _ = exact(T, X, Y, Z)
    return torch.stack([u, v, w]), {"method": "closed form (Ethier & Steinman 1994)"}, {}


def plot(problem, u, extras, figdir):
    k = problem.axes["z"].numel() // 2                               # the z = 0 plane
    x, y, t = problem.axes["x"], problem.axes["y"], problem.axes["t"]
    plotting.snapshot_rows(x, y, {n: u[i, :, :, :, k] for i, n in enumerate(("u", "v", "w"))}, t,
                           figdir / "reference.png", n=len(t), title="Track 10: Beltrami flow, z = 0 plane")
