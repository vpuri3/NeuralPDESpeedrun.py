"""Reference: published solution of the PirateNet / SOAP-PINN Grey-Scott benchmark (jaxpi
examples/grey_scott/data/grey_scott.mat; spectral solver, 101 x 200 x 200)."""
import numpy as np
import torch

from speedrun.external import load_mat
from tracks.data_common import plot_2d_time, save_problem_data

from .problem import B1, B2, C1, C2, EPS1, EPS2, TRACK

SOURCE = "grey_scott/data/grey_scott.mat"


def compute(device="cpu"):
    d = load_mat(SOURCE)
    got = [float(d[k].ravel()[0]) for k in ("b1", "b2", "c1", "c2", "ep1", "ep2")]
    assert np.allclose(got, [B1, B2, C1, C2, EPS1, EPS2]), got
    t, x, y = d["t"].ravel(), d["x"].ravel(), d["y"].ravel()
    U, V = d["usol"], d["vsol"]                       # [k, i(x), j(y)], as paired in the benchmark code
    save_problem_data(TRACK, t=t[1:], x=x, y=y, u0=U[0], v0=V[0])
    u = torch.from_numpy(np.stack([U[1:], V[1:]]))
    return u, {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plot_2d_time(problem, u, figdir, "Track 13: Grey-Scott")
