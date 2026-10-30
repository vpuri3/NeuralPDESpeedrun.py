"""Track 7B: lid-driven cavity (regularized lid), steady Navier-Stokes, Re = 1000."""
import torch

from tracks.cavity_common import make_cavity_problem

RE = 1000.0


def make_problem(device="cpu", dtype=torch.float64):
    return make_cavity_problem("track7B", RE, device=device, dtype=dtype)
