"""Reference: Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 at N = 8192 (8x the prescribed grid),
16 steps per snapshot interval; restricted to the prescribed 1024-point grid by injection.
Self-convergence is checked against N = 4096 with 8 steps per interval."""
import math

import torch

from speedrun import plotting
from speedrun.numerics.periodic import scalar_conservation_1d

from .problem import NX, make_problem


def run(p, N, substeps):
    nu = p.params["nu"]
    U = scalar_conservation_1d(p.fns["u0"], 2.0, N, lambda k: -nu * k**2, p.axes["t"], substeps,
                               dtype=p.dtype, device=p.device)
    return U[:, :: N // NX]


def compute(device="cpu"):
    p = make_problem(device=device)
    ref = run(p, 8192, 16)
    chk = run(p, 4096, 8)
    meta = {"method": "Fourier pseudo-spectral + ETDRK4, N=8192, dt=T/(Nt-1)/16",
            "check_N4096_vs_N8192_linf": float((chk - ref).abs().max()),
            "check_N4096_vs_N8192_rel_l2": float(torch.linalg.norm(chk - ref) / torch.linalg.norm(ref))}
    return ref[None], meta, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 5A: Burgers 1D, nu=1e-4")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", title="Track 5A snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 5A")


_ = math
