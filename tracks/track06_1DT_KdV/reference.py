"""Reference: published solution of the PirateNet / SOAP-PINN KdV benchmark (jaxpi
examples/kdv/data/kdv.mat; spectral solver, 201 x 512)."""
import numpy as np
import torch

from speedrun.external import load_mat
from tracks.data_common import plot_1d

from .problem import make_problem

SOURCE = "kdv/data/kdv.mat"


def compute(device="cpu"):
    d = load_mat(SOURCE)
    p = make_problem()
    assert np.allclose(d["t"].ravel()[1:], p.axes["t"].numpy()) and np.allclose(d["x"].ravel(), p.axes["x"].numpy())
    assert np.allclose(d["usol"][0], p.fns["u0"](p.axes["x"]).numpy())
    return torch.from_numpy(d["usol"][1:])[None], {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plot_1d(problem, u, figdir, "Track 06: KdV")
