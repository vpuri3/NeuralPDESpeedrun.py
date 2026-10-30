"""Reference: published solution of the PirateNet / SOAP-PINN chaotic Kuramoto-Sivashinsky benchmark
(jaxpi examples/ks_chaotic/data/ks_chaotic.mat; spectral solver, 251 x 512)."""
import torch

from speedrun.external import load_mat
from tracks.data_common import plot_1d, save_problem_data

from .problem import TRACK

SOURCE = "ks_chaotic/data/ks_chaotic.mat"


def compute(device="cpu"):
    d = load_mat(SOURCE)
    t, x, U = d["t"].ravel(), d["x"].ravel(), d["usol"]
    save_problem_data(TRACK, t=t[1:], x=x, u0=U[0])
    return torch.from_numpy(U[1:])[None], {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plot_1d(problem, u, figdir, "Track 05C: Kuramoto-Sivashinsky")
