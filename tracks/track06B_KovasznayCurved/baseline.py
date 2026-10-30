"""Record #0 (classical baseline): the spec's discretization (single-element Legendre PN-PN-2 on the
mapped quadrilateral, GLL quadrature, 3/2 dealiasing), N = 16, steady system solved by Newton (dense LU)."""
from tracks.kovasznay_common import sem_solve

N = 16


def warmup(problem):
    sem_solve(problem, 6)   # initializes torch.func (jacfwd/vmap); the N = 6 result is discarded


def solve(problem):
    return sem_solve(problem, N)
