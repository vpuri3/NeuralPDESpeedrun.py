"""Record #0 (classical baseline): single-element Legendre PN-PN-2, 3/2 dealiasing,
BDF3/EXT3 + incremental pressure projection, fast-diagonalization viscous and pressure solves), N = 128,
dt = 2 x (smallest GLL spacing)."""
from tracks.cavity_common import sem_solve

N = 128


def solve(problem, solver=None):
    return sem_solve(problem, N)
