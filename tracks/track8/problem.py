"""Track 8: steady Stokes flow in the regularized lid-driven cavity."""
import torch

from tracks.cavity_common import make_cavity_problem

RE = 0.0


def make_problem(device="cpu", dtype=torch.float64):
    return make_cavity_problem("track8", RE, device=device, dtype=dtype)
