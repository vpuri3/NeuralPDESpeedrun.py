"""Reference: published Ra = 1e6 solution of the PirateNet / SOAP-PINN Rayleigh-Taylor benchmark (jaxpi
examples/rayleigh_taylor/data/rayleigh_taylor_high_Ra.npy). As in the benchmark, the first 5 snapshots are
dropped and time is shifted to start at 0."""
import numpy as np
import torch

from speedrun.external import load_npy_dict
from tracks.data_common import plot_2d_time, save_problem_data

from .problem import TRACK

SOURCE = "rayleigh_taylor/data/rayleigh_taylor_high_Ra.npy"
START = 5


def compute(device="cpu"):
    d = load_npy_dict(SOURCE)
    c = d["coords"]
    x, y = np.unique(c[:, 0]), np.unique(c[:, 1])
    nx, ny = len(x), len(y)
    grid = lambda a: a.reshape(*a.shape[:-1], ny, nx).swapaxes(-1, -2)   # x-fastest points -> [.., i(x), j(y)]  # noqa: E731
    vel, T, p = d["velocity"][START:], d["temperature"][START:], d["pressure"][START:]
    U, V, T, P = grid(vel[..., 0]), grid(vel[..., 1]), grid(T), grid(p)
    t = d["t"][START:] - d["t"][START]
    save_problem_data(TRACK, t=t[1:], x=x, y=y, u0=U[0], v0=V[0], p0=P[0], T0=T[0], Ra=d["Ra"], Pr=d["Pr"],
                      a1=d["alpha1"], a2=d["alpha2"], a4=d["alpha4"])
    u = torch.from_numpy(np.stack([U[1:], V[1:], T[1:]]).astype(np.float64))
    return u, {"method": "published data (jaxpi)", "source": SOURCE}, {}


def plot(problem, u, extras, figdir):
    plot_2d_time(problem, u, figdir, "Track 15: Rayleigh-Taylor", names=("u", "v", "T"),
                 cmaps={"T": "inferno"})
