"""Record #0 (classical baseline): the spec's discretization (single-element Legendre PN-PN-2), N = 64,
steady Stokes by Uzawa CG on the pressure (inverse pressure-mass preconditioner), fast-diagonalization
viscous solves."""
from tracks.cavity_common import sem_solve

N = 64


def solve(problem):
    return sem_solve(problem, N)
