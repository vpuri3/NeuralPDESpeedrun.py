"""Reference: Kovasznay's closed-form solution at the mapped points (cross-checked with the
PN-PN-2 solver on the Gordon-Hall-mapped element at N = 24)."""
import torch

from tracks.kovasznay_common import plot as _plot
from tracks.kovasznay_common import sem_solve, u_bc, v_bc

from .problem import make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    R, S = p.grid()
    X, Y = p.fns["map"](R, S)
    u = torch.stack([u_bc(X, Y), v_bc(X, Y)])
    chk = float((sem_solve(p, 24) - u).abs().max())
    return u, {"method": "closed form (Kovasznay 1948) at the mapped points", "check_sem_N24_linf": chk}, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir, "Track 06B: Kovasznay flow, Re = 40, curved trapezoid")
