"""Reference: the spec's single-element Legendre PN-PN-2 discretization at N = 384, BDF3/EXT3 with
incremental pressure projection, dt = 2 x (smallest GLL spacing) (self-convergence checked against N = 256;
the time-step error at this CFL is ~1e-6, see PLAN.md)."""
from tracks.cavity_common import plot as _plot
from tracks.cavity_common import reference

from .problem import make_problem


def compute(device="cpu"):
    u, meta = reference(make_problem(device=device), 384, 256)
    return u, meta, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir)
