"""Track 7A: lid-driven cavity (regularized lid), steady Navier-Stokes, Re = 100."""
import torch

from tracks.cavity_common import make_cavity_problem

RE = 100.0


def make_problem(device="cpu", dtype=torch.float64):
    return make_cavity_problem("track7A", RE, device=device, dtype=dtype)
