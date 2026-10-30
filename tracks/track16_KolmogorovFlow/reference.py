"""Reference: published Re = 1e4 solution of the PirateNet / SOAP-PINN Kolmogorov-flow benchmark (jaxpi
examples/kolmogorov_flow/data/kolmogorov_flow_Re10000_256.npy, float32). Time is shifted to start at 0."""
import numpy as np
import torch

from speedrun.external import load_npy_dict
from tracks.data_common import plot_2d_time, save_problem_data

from .problem import NU, TRACK

SOURCE = "kolmogorov_flow/data/kolmogorov_flow_Re10000_256.npy"


def compute(device="cpu"):
    d = load_npy_dict(SOURCE)
    assert np.isclose(float(d["nu"]), NU)
    c = d["coords"].astype(np.float64)
    x, y = np.unique(c[:, 0]), np.unique(c[:, 1])
    nx, ny = len(x), len(y)
    grid = lambda a: a.astype(np.float64).reshape(*a.shape[:-1], ny, nx).swapaxes(-1, -2)  # noqa: E731
    U, V = grid(d["velocity"][..., 0]), grid(d["velocity"][..., 1])
    t = d["t"].astype(np.float64) - float(d["t"][0])
    save_problem_data(TRACK, t=t[1:], x=x, y=y, u0=U[0], v0=V[0])
    return torch.from_numpy(np.stack([U[1:], V[1:]])), {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plot_2d_time(problem, u, figdir, "Track 16: Kolmogorov flow", vorticity=True)
