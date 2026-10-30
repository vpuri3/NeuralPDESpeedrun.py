"""Track 7A: lid-driven cavity [-1, 1]^2, u = 1 lid, Re = 100, impulsive start, t in [0, 5] (spec `LDC` case)."""
import torch

from tracks.cavity_common import make_cavity_problem

RE = 100.0


def make_problem(device="cpu", dtype=torch.float64):
    return make_cavity_problem("track07A_CavityRe100", RE, device=device, dtype=dtype)
