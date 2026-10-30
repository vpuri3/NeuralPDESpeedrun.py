"""Reference: the spec's single-element Legendre PN-PN-2 discretization at N = 512, Uzawa CG
(self-convergence checked against N = 384)."""
from tracks.cavity_common import plot as _plot
from tracks.cavity_common import reference

from .problem import make_problem


def compute(device="cpu"):
    u, meta = reference(make_problem(device=device), 512, 384)
    return u, meta, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir)
