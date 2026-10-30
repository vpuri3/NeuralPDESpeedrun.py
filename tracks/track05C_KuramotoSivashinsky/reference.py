"""Reference: Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 at N = 1024 (4x the prescribed grid),
4 steps per snapshot interval; restricted to the prescribed grid by injection.
Self-convergence is checked against N = 512 with 2 steps per interval."""
import math

import torch

from speedrun import plotting
from speedrun.numerics.periodic import scalar_conservation_1d

from .problem import NX, make_problem


def run(p, N, substeps):
    nu = p.params["nu"]
    u0 = lambda s: p.fns["u0"](s - math.pi)  # noqa: E731  (grid offset: x = -pi + s)
    U = scalar_conservation_1d(u0, 2 * math.pi, N, lambda k: k**2 - nu * k**4, p.axes["t"], substeps,
                               dtype=p.dtype, device=p.device)
    return U[:, :: N // NX]


def compute(device="cpu"):
    p = make_problem(device=device)
    ref = run(p, 1024, 4)
    chk = run(p, 512, 2)
    meta = {"method": "Fourier pseudo-spectral + ETDRK4, N=1024, dt=T/(Nt-1)/4",
            "check_N512_vs_N1024_linf": float((chk - ref).abs().max()),
            "check_N512_vs_N1024_rel_l2": float(torch.linalg.norm(chk - ref) / torch.linalg.norm(ref))}
    return ref[None], meta, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 5C: Kuramoto-Sivashinsky")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", title="Track 5C snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 5C")
