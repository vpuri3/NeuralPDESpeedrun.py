"""Record #0 (classical baseline): single-element Legendre PN-PN-2), N = 192,
steady Stokes by Uzawa CG on the pressure (inverse pressure-mass preconditioner), fast-diagonalization
viscous solves."""
from tracks.cavity_common import sem_solve

N = 192


def solve(problem, solver=None):
    return sem_solve(problem, N)
