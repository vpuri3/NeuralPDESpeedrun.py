"""Track 07B: lid-driven cavity [-1, 1]^2, u = 1 lid, Re = 1000, impulsive start, t in [0, 5]."""
import torch

from tracks.cavity_common import make_cavity_problem

RE = 1000.0


def make_problem(device="cpu", dtype=torch.float64):
    return make_cavity_problem("track07B_CavityRe1000", RE, device=device, dtype=dtype)
