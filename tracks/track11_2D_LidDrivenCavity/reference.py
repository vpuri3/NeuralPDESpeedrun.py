"""Reference: published Re = 5000 solution of the PirateNet / SOAP-PINN lid-driven cavity benchmark (jaxpi
examples/ldc/data/ldc_Re5000.mat; classical solver, 256 x 256)."""
import numpy as np
import torch

from speedrun import plotting
from speedrun.external import load_mat

from .problem import RE, make_problem

SOURCE = "ldc/data/ldc_Re5000.mat"


def compute(device="cpu"):
    d = load_mat(SOURCE)
    p = make_problem()
    assert np.isclose(float(d["nu"].ravel()[0]), 1.0 / RE)
    assert np.allclose(d["x"].ravel(), p.axes["x"].numpy()) and np.allclose(d["y"].ravel(), p.axes["y"].numpy())
    u = torch.from_numpy(np.stack([d["u"], d["v"]]))                 # [i(x), j(y)]
    return u, {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plotting.cavity(problem.axes["x"], problem.axes["y"], u[0], u[1], figdir / "reference.png",
                    title="Track 11: lid-driven cavity, Re = 5000")
