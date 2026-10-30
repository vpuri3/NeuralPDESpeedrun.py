"""Reference: Chebyshev collocation of the streamfunction formulation at N = 48
(self-convergence checked against N = 32)."""
from tracks.cavity_common import plot as _plot
from tracks.cavity_common import reference

from .problem import make_problem


def compute(device="cpu"):
    u, meta = reference(make_problem(device=device), 48, 32)
    return u, meta, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir)
