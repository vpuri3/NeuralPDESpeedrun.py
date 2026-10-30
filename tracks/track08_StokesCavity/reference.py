"""Reference: single-element Legendre PN-PN-2 discretization at N = 768, Uzawa CG
(self-convergence checked against N = 512; N = 1024 is out of reach: the pressure mass matrix is no longer
numerically positive definite in fp64)."""
from tracks.cavity_common import plot as _plot
from tracks.cavity_common import reference

from .problem import make_problem


def compute(device="cpu"):
    u, meta = reference(make_problem(device=device), 768, 512)
    return u, meta, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir)
