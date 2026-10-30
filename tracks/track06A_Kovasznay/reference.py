"""Reference: Kovasznay's closed-form solution (cross-checked with the PN-PN-2 spectral solver at N = 24)."""
import torch

from tracks.kovasznay_common import plot as _plot
from tracks.kovasznay_common import sem_solve, u_bc, v_bc

from .problem import X0, X1, Y0, Y1, make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    X, Y = p.grid()
    u = torch.stack([u_bc(X, Y), v_bc(X, Y)])
    p.axes = {"r": 2 * (p.axes["x"] - X0) / (X1 - X0) - 1, "s": 2 * (p.axes["y"] - Y0) / (Y1 - Y0) - 1}
    chk = float((sem_solve(p, 24) - u).abs().max())
    return u, {"method": "closed form (Kovasznay 1948)", "check_sem_N24_linf": chk}, {}


def plot(problem, u, extras, figdir):
    _plot(problem, u, figdir, "Track 06A: Kovasznay flow, Re = 40")
